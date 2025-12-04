from fastchat.model import get_conversation_template


def load_conversation_template(template_name):
    conv_template = get_conversation_template(template_name)
    if conv_template.name == 'zero_shot':
        conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.sep = '\n'
    elif conv_template.name == 'llama-2':
        conv_template.sep2 = conv_template.sep2.strip()
    
    return conv_template


def generate_input(tokenizer,prompt,adv_suffix=None):   
    messages = [{"role": "user", "content": prompt}]

    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    ) + " "
    
    return formatted_prompt