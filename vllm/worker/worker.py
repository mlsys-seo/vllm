"""A GPU worker class."""
import os
from typing import Dict, List, Tuple, Optional

import torch
import torch.distributed

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.model_executor import get_model, InputMetadata, set_random_seed
from vllm.model_executor.parallel_utils.parallel_state import (
    initialize_model_parallel)
from vllm.sampling_params import SamplingParams
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
from vllm.worker.cache_engine import CacheEngine
from vllm.utils import get_gpu_memory

import numpy as np


class Worker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        rank: Optional[int] = None,
        distributed_init_method: Optional[str] = None,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.rank = rank
        self.distributed_init_method = distributed_init_method

        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        self.cache_config = None
        self.block_size = None
        self.cache_engine = None
        self.cache_events = None
        self.gpu_cache = None

    def init_model(self):
        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        # Env vars will be set by Ray.
        self.rank = self.rank if self.rank is not None else int(
            os.getenv("RANK", "-1"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{local_rank}")
        if self.rank < 0:
            raise ValueError("Invalid or unspecified rank.")
        torch.cuda.set_device(self.device)

        # Initialize the distributed environment.
        _init_distributed_environment(self.parallel_config, self.rank,
                                      self.distributed_init_method)

        # Initialize the model.
        set_random_seed(self.model_config.seed)
        self.model = get_model(self.model_config)

    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
    ) -> Tuple[int, int]:
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.

        # Enable top-k sampling to reflect the accurate memory usage.
        vocab_size = self.model.config.vocab_size
        sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        seqs = []
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            seq_data = SequenceData([0] * seq_len)
            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
            )
            seqs.append(seq)

        input_tokens, input_positions, input_metadata = self._prepare_inputs(
            seqs)

        # Execute the model.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        self.model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=[(None, None)] * num_layers,
            input_metadata=input_metadata,
            cache_events=None,
        )

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        total_gpu_memory = get_gpu_memory()
        cache_block_size = CacheEngine.get_cache_block_size(
            block_size, self.model_config, self.parallel_config)
        num_gpu_blocks = int(
            (total_gpu_memory * gpu_memory_utilization - peak_memory) //
            cache_block_size)
        num_cpu_blocks = int(cpu_swap_space // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        torch.cuda.empty_cache()

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)
        return num_gpu_blocks, num_cpu_blocks

    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        self.cache_config = cache_config
        self.block_size = cache_config.block_size
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config)
        self.cache_events = self.cache_engine.events
        self.gpu_cache = self.cache_engine.gpu_cache

    def _prepare_inputs(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []

        # Add prompt tokens.
        prompt_lens: List[int] = []
        for seq_group_metadata in seq_group_metadata_list:
            if not seq_group_metadata.is_prompt:
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            # Use any sequence in the group.
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)

            input_tokens.extend(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.extend(range(len(prompt_tokens)))

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.extend([0] * prompt_len)
                continue

            # Compute the slot mapping.
            block_table = seq_group_metadata.block_tables[seq_id]
            for i in range(prompt_len):
                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        # Add generation tokens.
        max_context_len = 0
        max_num_blocks_per_seq = 0
        context_lens: List[int] = []
        generation_block_tables: List[List[int]] = []
        for seq_group_metadata in seq_group_metadata_list:
            if seq_group_metadata.is_prompt:
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)

                context_len = seq_data.get_len()
                position = context_len - 1
                input_positions.append(position)

                block_table = seq_group_metadata.block_tables[seq_id]
                generation_block_tables.append(block_table)

                max_context_len = max(max_context_len, context_len)
                max_num_blocks_per_seq = max(max_num_blocks_per_seq,
                                             len(block_table))
                context_lens.append(context_len)

                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        # Optimization: Pad the input length to be a multiple of 8.
        # This is required for utilizing the Tensor Cores in NVIDIA GPUs.
        input_tokens = _pad_to_alignment(input_tokens, multiple_of=8)
        input_positions = _pad_to_alignment(input_positions, multiple_of=8)

        # Convert to tensors.
        tokens_tensor = torch.cuda.LongTensor(input_tokens)
        positions_tensor = torch.cuda.LongTensor(input_positions)
        slot_mapping_tensor = torch.cuda.IntTensor(slot_mapping)
        context_lens_tensor = torch.cuda.IntTensor(context_lens)
        padded_block_tables = [
            _pad_to_max(block_table, max_num_blocks_per_seq)
            for block_table in generation_block_tables
        ]
        block_tables_tensor = torch.cuda.IntTensor(padded_block_tables)

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        input_metadata = InputMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            slot_mapping=slot_mapping_tensor,
            context_lens=context_lens_tensor,
            max_context_len=max_context_len,
            block_tables=block_tables_tensor,
        )
        return tokens_tensor, positions_tensor, input_metadata

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> SamplerOutput:
        # Issue cache operations.
        issued_cache_op = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True

        if issued_cache_op:
            cache_events = self.cache_events
        else:
            cache_events = None

        # If there is no input, we don't need to execute the model.
        if not seq_group_metadata_list:
            if cache_events is not None:
                for event in cache_events:
                    event.wait()
            return {}

        # Prepare input tensors.
        input_tokens, input_positions, input_metadata = self._prepare_inputs(
            seq_group_metadata_list)

        # Execute the model.
        output = self.model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=self.gpu_cache,
            input_metadata=input_metadata,
            cache_events=cache_events,
        )

        if len(input_metadata.block_tables) > 0:
            # 첫 째 dim이 prompt
            if len(input_metadata.block_tables[0]) - len(input_metadata.quantized) \
                == 2:
                
                # start
                # TODO: ALL Prompt
                # block_tables List[List] peomprr 수개수ㅋ만 큼 ㄷ있ㄷ
                block_table_0 = input_metadata.block_tables[0]
                # quantized -> scale ㄱ변ㄱ 저예저
                target_idx = block_table_0[len(input_metadata.quantized)]

                self._quantize(target_idx)
 
        return output

    def _quantize(
        self,
        target_idx: int,
    )-> torch.tensor:
        # TODO: k 
        kv = 1 # v=1
        for layer in range(len(self.gpu_cache)):         
            # target_tensor: 1 physical block
            target_tensor = self.gpu_cache[layer][kv][target_idx]
            # num_kv_heads, HEAD_SIZE, BLOCJ_SIZE
            # num_heads, num_elements, num_tokens
            
            TARGET_BIT = 4
            n = 2 ** (TARGET_BIT - 1)
            
            scale = torch.max(target_tensor.max().abs(), target_tensor.min().abs())
            scale = torch.clamp(scale, min=1e-8) / n
            # zero_point = torch.tensor(0.0).to(scale.device)
            quantized_tensor = target_tensor.clone()
            
            # TODO: rounding_mode 확인
            quantized_tensor.div_(scale, rounding_mode="trunc")
            quantized_tensor = quantized_tensor.type(torch.int16)
            
            # quant dequnt test
            '''
            # TEST
            quantized_tensor = quantized_tensor.type(torch.float16)
            quantized_tensor.mul_(scale)
        
            print(f"scale: {scale}")
            print(target_tensor[1][2])
            print(quantized_tensor[1][2])
            
            >>> a
            [-0.0715, -0.1412, 0.0081, -0.2654, -0.0616, 0.0281, -0.1429, 0.0033, -0.283, -0.2241, 0.0311, -0.2737, -0.2747, 0.0388, -0.1098, 0.0626]
            >>> e
            [0.0, 0.0, 0.0, -0.17236328125, 0.0, 0.0, 0.0, 0.0, -0.17236328125, -0.17236328125, 0.0, -0.17236328125, -0.17236328125, 0.0, 0.0, 0.0]
            # python 에서 같은 연산 똑같이 해봄
            '''
            
            # bit mask test
            '''
            #TEST
            before = quantized_tensor.clone()
            
            # packing
            and_val = torch.tensor(0xf, dtype=torch.int16).to(quantized_tensor.device)
            quantized_tensor = torch.bitwise_and(quantized_tensor, and_val)
            
            if layer == 4:
                # TEST
                before_list = before[2][4]
                after_list = quantized_tensor[2][4]
                
                import pdb; pdb.set_trace()
                
            (Pdb) p before_list
            tensor([ 0,  0,  0,  0, -1,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
                device='cuda:0', dtype=torch.int16)
            (Pdb) p after_list
            tensor([ 0,  0,  0,  0, 15,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
                device='cuda:0', dtype=torch.int16)
            '''
                            
            # packing
            and_val = torch.tensor(0xf, dtype=torch.int16).to(quantized_tensor.device)
            quantized_tensor = torch.bitwise_and(quantized_tensor, and_val)
            
            num_heads = len(quantized_tensor)
            num_elems = len(quantized_tensor[0]) # HEAD_SIZE
            num_tokens = len(quantized_tensor[0][0]) # BLOCK_SIZE
            
            for head_idx in range(num_heads):
                for elem_idx in range(num_elems):
                    for token_idx in range(num_tokens):
                        write_idx = token_idx // 4
                        write_offset = token_idx % 4
                        
                        read = quantized_tensor[head_idx][elem_idx][token_idx]
                        read <<= 4 * (3 - write_offset)
                        if write_offset == 0:
                            quantized_tensor[head_idx][elem_idx][write_idx] = torch.tensor(0, dtype=torch.int16, device=quantized_tensor.device)
                        quantized_tensor[head_idx][elem_idx][write_idx] |= read
                        
            '''
            if layer == 5:
                a = 0
                for i in before[2][3]:
                    if (a==4):
                        break
                    print(f"{a}: {bin(i)}")
                    a += 1
                print(bin(quantized_tensor[2][3][0]))
                input()
                
            0: 0b0
            1: 0b1
            2: 0b10
            3: 0b0
            0b 1 0010 0000
            '''

            
            # cpoy to gpu_cache
            self.gpu_cache[layer][kv][target_idx] = quantized_tensor.view(dtype=torch.float16)
            
            '''
            # TEST
            if layer == 6:
                target_list_q = quantized_tensor[5][3]
                target_list_c = self.gpu_cache[layer][kv][target_idx][5][3].view(dtype=torch.int16)
                for i in range(len(target_list_q)):
                    print(i)
                    print(bin(target_list_q[i]))
                    print(bin(target_list_c[i]))
                    print()
            
                input()
                
            0
            0b111100000000
            0b111100000000

            1
            -0b10000000000000
            -0b10000000000000

            2
            0b0
            0b0

            3
            0b111100000000
            0b111100000000

            4
            -0b10000000000000
            -0b10000000000000

            5
            0b0
            0b0

            6
            0b0
            0b0

            7
            0b0
            0b0

            8
            0b0
            0b0
            '''
                
            # update scale list
        return scale

def _init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
) -> None:
    """Initialize the distributed environment."""
    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size "
                f"({torch_world_size} vs. {parallel_config.world_size}).")
    elif not distributed_init_method:
        raise ValueError(
            "distributed_init_method must be set if torch.distributed "
            "is not already initialized")
    else:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    initialize_model_parallel(parallel_config.tensor_parallel_size,
                              parallel_config.pipeline_parallel_size)


def _pad_to_alignment(x: List[int], multiple_of: int) -> List[int]:
    return x + [0] * ((-len(x)) % multiple_of)


def _pad_to_max(x: List[int], max_len: int) -> List[int]:
    return x + [0] * (max_len - len(x))