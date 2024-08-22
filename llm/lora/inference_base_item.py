import os

os.environ["WANDB_DISABLED"] = "true"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
import pickle
from random import randrange
from datasets import load_from_disk
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer

class ChatBot:
    # def __init__(self, model_path):
    #     self.model = AutoPeftModelForCausalLM.from_pretrained(
    #         model_path,
    #         low_cpu_mem_usage=True,
    #         torch_dtype=torch.float16,
    #         load_in_4bit=True,
    #     )
    #     self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    def __init__(self, model_path):
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            load_in_4bit=True,
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)

    def get_response(self, message):
        input_ids = self.tokenizer(message, return_tensors="pt", truncation=True).input_ids.cuda()
        # print(input_ids.shape)
        outputs = self.model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9,temperature=0.5)
        # print(outputs.shape)
        return self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(message):]

    def chat(self, nb_item_list, item_info_dict, item_profile_dict, iid, result_dict):
        content_begin = "<s>[INST]Now you are a profile generator. I will provide you with textual information about one target item as well as a list of other items with their textual information and their targeting users' profiles. Based on this information, please generate the user profile of this target item."
        content_end = "Please provide the profile with 5 identities and 5 interests strictly in the following format: User identity: [Identity 1], [Identity 2], [Identity 3], [Identity 4], [Identity 5]; User interests: [Interest 1], [Interest 2], [Interest 3], [Interest 4], [Interest 5].[/INST]"

        content = "The target item text information: "
        # mind
        # content += '[cate: '
        # content += item_info_dict[iid]['cate']
        # content += ', sub_cate: '
        # content += item_info_dict[iid]['sub_cate']
        # content += ', title: '
        # content += item_info_dict[iid]['title']
        # content += ', abstract: '
        # content += item_info_dict[iid]['abstract']
        # netflix
        # content += '[title: '
        # content += item_info_dict[iid]['title']
        # content += ', year: '
        # content += str(item_info_dict[iid]['year'])
        content += '[title: '
        content += item_info_dict[iid]
        content += ']. Here is the list of item (text information and the profile of the users that the item is targeting):\n'
        for iid_ in nb_item_list:
            content += 'item '
            content += str(iid_)
            content += ': ['
            content += 'Text information: '
            # mind
            # content += 'cate: '
            # content += item_info_dict[iid_]['cate']
            # content += ', sub_cate: '
            # content += item_info_dict[iid_]['sub_cate']
            # content += ', title: '
            # content += item_info_dict[iid_]['title']
            # netflix
            # content += 'title: '
            # content += item_info_dict[iid]['title']
            # content += ', year: '
            # content += str(item_info_dict[iid]['year'])
            content += 'title: '
            content += item_info_dict[iid]
            content += ', the profile of the users that the item is targeting: '
            content += item_profile_dict[iid_]['output']
            content += '];\n'

        content = content_begin + content + content_end

        # print("SEND: ")
        # print(content)

        response = self.get_response(content)
        result_dict[iid] = response

        # print("GPT: ")
        # print(response)
        # print()


file_self_instruction_dict_item = './../../data/netflix/self_instruction_dict_item_full.pkl'
with open(file_self_instruction_dict_item, 'rb') as fs:
    self_instruction_dict_item = pickle.load(fs)

file_item_info_dict = './../../data/netflix/item_info_dict.pkl'
with open(file_item_info_dict, 'rb') as fs:
    item_info_dict = pickle.load(fs)

file_item_profile_dict = './../../data/netflix/gpt_output_dict_item_side_filter.pkl'
with open(file_item_profile_dict, 'rb') as fs:
    item_profile_dict = pickle.load(fs)

result_dict = {}
chatbot = ChatBot("./../ft_models/llama_lora_netflix_item_v0/merged_model_928")

counter = 0

print(len(self_instruction_dict_item))

# for uid in user_item_dict:
#     item_list = user_item_dict[uid]
#     chatbot.chat(item_list, item_info_dict, uid, result_dict)
#     print(counter)
#     counter += 1

for iid in self_instruction_dict_item:
    if counter < 220000:
        counter += 1
        continue
    # if counter >= 220000:
    #     counter += 1
    #     continue
    nb_item_list = self_instruction_dict_item[iid]
    chatbot.chat(nb_item_list, item_info_dict, item_profile_dict, iid, result_dict)
    print(counter)
    counter += 1

print(counter)

output_f = open('./../../data/netflix/item_profile/llama_lora_netflix_item_v0/item_profile_dict_220000_end.pkl', 'wb')
pickle.dump(result_dict, output_f)
output_f.close()





