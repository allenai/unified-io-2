import logging
import re

import gin

from t5x.examples.unified_io.data.prompt_dict import PROMPT_DICT


@gin.configurable
class Prompt:
    """Configurable interface for getting prompts"""

    def __init__(self, original_flag=True, revised_original_flag=False, manual_flag=True,
                 gpt3_flag=True, single_prompt=False, dbg=None):
        self.prompt_list = []
        self.original_flag = original_flag
        self.revised_original_flag = revised_original_flag
        self.manual_flag = manual_flag
        self.gpt3_flag = gpt3_flag
        self.single_prompt = single_prompt
        self.dbg = dbg

    def get_prompt_list(self, task_name, dataset_name):
        if self.dbg:
            logging.info(f"Using dbg prmopt {self.dbg}")
            return [self.dbg]
        prompt_list = []
        if self.original_flag:
            if self.revised_original_flag and 'revised_original' in PROMPT_DICT[task_name]:
                prompt_list += PROMPT_DICT[task_name]['revised_original']
            else:
                prompt_list += PROMPT_DICT[task_name]['original']
            if self.revised_original_flag and 'revised_original' in PROMPT_DICT[dataset_name]:
                prompt_list += PROMPT_DICT[dataset_name]['revised_original']
            else:
                prompt_list += PROMPT_DICT[dataset_name]['original']
        if self.manual_flag:
            if 'manual' in PROMPT_DICT[task_name]:
                prompt_list += PROMPT_DICT[task_name]['manual']
            if 'manual' in PROMPT_DICT[dataset_name]:
                prompt_list += PROMPT_DICT[dataset_name]['manual']
        if self.gpt3_flag:
            if 'gpt3' in PROMPT_DICT[task_name]:
                prompt_list += PROMPT_DICT[task_name]['gpt3']
                
            if 'gpt3' in PROMPT_DICT[dataset_name]:            
                prompt_list += PROMPT_DICT[dataset_name]['gpt3']
        if not prompt_list:
            raise ValueError(f"No prompts for {task_name}/{dataset_name}")
        if self.single_prompt:
            logging.info(f"Using prompt \"{prompt_list[0]}\" for {task_name} {dataset_name}")
            return prompt_list[:1]
        return prompt_list

