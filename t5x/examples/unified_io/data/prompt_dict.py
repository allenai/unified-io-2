"""All prompts"""

from collections import defaultdict
from typing import Optional, List

from dataclasses import dataclass, replace

PROMPT_DICT = defaultdict(lambda: dict(original=[], manual=[], gpt3=[], template=[]))


ENTAILMENT_LABELS = ['entailment', 'neutral', 'contradiction']
TRUNCATE = "[TR]"


@dataclass
class NlpPrompt:
    prompt: str
    class_names: Optional[List] = None
    list_mode: Optional[str] = None


def _original(question, options, one_sentence=False):
    if one_sentence:
        context = "sentence1: {sentence1}"
    else:
        context = "sentence1: {sentence1} sentence2: {sentence2}"
    return NlpPrompt(f"{context} question: {question}", options)


# Task-specific prompts
# =====================
PROMPT_DICT['NLP MC Question'] = dict(
    original=["Question: {question} Answer:"],
    manual=[],
)

PROMPT_DICT['NLP Instruction Context'] = dict(
  original=[f"Instruction: {{instruction}} Context: {{context}}{TRUNCATE}"],
  manual=[
    f"Context: {{context}}{TRUNCATE} Instruction: {{instruction}}",
    f"Context: {{context}}{TRUNCATE} Query: {{instruction}} Response:",
    f"Context: {{context}}{TRUNCATE}\nInstruction: {{instruction}}\nOutput:",
    "Follow the instructions: {instruction}\nContext: {context}",
    "{instruction}\nUse the information to help generate a response: {context}",
    f"{{context}}{TRUNCATE}\n\n{{instruction}}\n\n",
    "Please read these instructions carefully and return a response without any additional content.\n\n"
    "{instruction}\n\nAdditional context: {context}\n",
    f"Instruction: {{instruction}} Context: {{context}}{TRUNCATE}",
    f"{{context}}{TRUNCATE} {{instruction}}\n",
    f"{{context}}{TRUNCATE} {{instruction}}",
    f"{{instruction}}\n{{context}}{TRUNCATE}\n",
    f"{{instruction}} {{context}}{TRUNCATE}",
    f"Please help me with this. {{instruction}}\n{{context}}\n",
  ],
)

PROMPT_DICT['NLP Instruction'] = dict(
    original=["{instruction}"],
    manual=[
        "Query: {instruction}\nResponse:",
        "Instruction: {instruction}\nOutput:",
        "Request: {instruction}\n",
        "Please read this carefully and provide a helpful response: {instruction}",
        "You are a helpful assistant responding to a user query, provide a correct and "
        "useful response\nQuery: {instruction}",
        "{instruction}\n",
        "{instruction}. Think carefully and return a compact, accurate response.",
        "Below is a user request, respond to the best of your ability:\n{instruction}",
        "Help me with this. {instruction}",
        "Please assist me. {instruction}",
    ],
)

PROMPT_DICT['gms8k']['manual'] = [
    'Please provide a short answer. Question: {}',
    '{} Answer: ',
    'Problem: {}\nFinal Answer: ',
    'Problem: {} Answer with the final result only.',
    'What is the final result of the math problem: {}',
    'In short terms, solve: {}',
]

PROMPT_DICT['gms8k_explain']['manual'] = [
    '{}',   # have chance w/o instruction --- question only
    'Provide the answer with step-by-step explanations to the following question: {}',
    'Answer the following question and provide thorough explanations: {}',
    '{} Walk through your solution.',
    "Can you elaborate on how you'd tackle the problem:\n{}",
    "Break down the process for the problem: {}. I'd like to understand each step.",
    'For the mathematical challenge "{}", explain your reasoning.',
    "Thinking out loud, how would you process: {}",
    "Problem: {}\nTask: answer the question with detailed explanations.",
    "Imagine you are a graduate student. Provide a comprehensive breakdown for the problem: {}",
    "Imagine you're a graduate student. Solve the following problem and outline your processes: {}",
]

PROMPT_DICT['gms8k_socratic_explain']['manual'] = [
    'Provide the answer with socratic walk through to the following question: \n{}\nAnswer format: "question ** answer"',
    'Answer the query:\n{}. Ensure you structure your solution as self-posed questions followed by answers, using the " ** " format.',
    'For the question "{}", break down the solution using a dialogue of questions you\'d ask yourself and their corresponding answers. Format it with "{sub-question} ** {sub-answer}".',
    'For the problem: {}, can you elucidate your thought process as a series of self-directed questions and answers? Use the "**" marker between each.',
    "Imagine you're introspecting. Walk me through your thought process for the problem: '{}' by posing questions to yourself and responding.",
    "Question: {}\nTask: Detail them as a series of questions you'd pose to yourself and their answers, using the '**' delineation.",
    'For the challenge: "{}", could you lay out your thought process as a sequence of self-questions and answers? Use "**" as the separator.',
    'Given the problem: {}, explain your methodology in a self-inquisitive manner, separating each question from its answer with "**".',
    "How would you approach the problem: {} Respond as if you're questioning and answering yourself, using \"**\" between each self-question and answer.",
    'Problem: {}\nTask: Provide answer with detailed explanations.\nConstraint: Format the answer in a socratic manner, separating each self-inquisitive question from its answer with "**"',
]


PROMPT_DICT["Multiple_Instance_Image_Classification"] = dict(
    original=[
        "Build a comma separated list of each type of object that appears in the image."
    ],
    manual=[]
)


PROMPT_DICT['Multi_Refexp_Segmentation'] = dict(
    original=[
        'Segment any objects referred to by the expression: {refexp}.'
    ],
    manual=[
        'Segment the {refexp}.',
        "Segment any object that {refexp} matches",
        "Find the object that match \"{refexp}\" and then segment them",
        "Generate an image with white pixels for parts of the image matches {refexp} and white pixels elsewhere.",
        "Build a black-and-white image that shows pixels matching \"{refexp}\".",
        'Expression: {refexp}. Instructions: What pixels does the expression refer to?',
        'Build a binary mask showing which pixels are referred to by {refexp}.',
        'The text \"{refexp}\" refers to 0 or more objects in the image, segment those objects.',
        "Show me what pixels {refexp} refers to.",
        "What pixels are part of {refexp}?",
    ]
)



PROMPT_DICT['Refexp_Segmentation'] = dict(
    original=[
        'Segment the object referred to by the expression: {refexp}.'
    ],
    manual=PROMPT_DICT['Multi_Refexp_Segmentation']["manual"] + [
        "An object in this image matches {refexp}, segment it",
    ]
)


PROMPT_DICT['Multi_Refexp'] = dict(
    original=[
        'Return the bounding boxes of any objects referred to by the expression: {description}.'
    ],
    manual=[]
)


PROMPT_DICT['Refexp Augmentented'] = dict(
    original=[
        'Output {property} of the region in the image described by: {refexp}'
    ],
    manual=[
        'What is {property} of the region {refexp} refers to?',
        'If there was a rectangle around {refexp}, what would {property} be?',
        'Expression: {refexp} Instruction: What is {property} of the region the expression refers to?',
        'For the region in the image that matches the expression "{refexp}", return {property}',
        'In the image there is a region containing: {refexp}, return {property}',
        'Instruction: Compute {property} of that image region that matches the description. Description: {refexp}',
        'Figure out what bounding box of the region in the images matches {refexp}, and then return {property}',
    ]
)


PROMPT_DICT['Refexp Augmentented Two Refs'] = dict(
    original=[
        ['Output {property} of the image region described by {refexp1}, and the region described by {refexp2}',
         "{target1} and {target2}"]
    ],
    manual=[
        ['State the {property} of the image region that matches 1: {refexp1} and 2: {refexp2}',
         "1: {target1} 2: {target2}"],
        ['For the bounding boxes that contain \"{refexp1}\" and \"{refexp2}\", return {property}',
         "{target1} and {target2}"],
        ['Expression 1: {refexp1} Expression 2: {refexp2} '
         'Instruction: return {property} of the image regions that match expression 1 and expression 2.',
         "Expression 1: {target1} Expression 2: {target2}"],
        ['Expression 1: {refexp1} Expression 2: {refexp2} Instruction: return {property} of the image region that matches expression 2.',
         "{target2}"],
        ['Expression 1: {refexp1} Expression 2: {refexp2} Instruction: return {property} of the image region that matches expression 1.',
         "{target1}"],
        ['State {property} of the image bounding boxes that match {refexp1} and {refexp2}',
         "{target1} and {target2}"],
        ['What is {property} of the image region {refexp1} refers to? What about the region {refexp2} refers to?',
         "{target1}. {target2}."],
        ['Imagine there is a rectangle around "{refexp1}" what is {property}? What about for "{refexp2}"?',
         "{target1}. {target2}."]
    ]
)


PROMPT_DICT['Refexp Augmentented Pairs'] = dict(
    original=[
        ['Output {property1} of the image region described by {refexp1}, and then {property2} described by {refexp2}',
         "{target1} and {target2}"]
    ],
    manual=[
        ['State 1: the {property1} of the image region that matches {refexp1} and 2: the {property2} of the region that matches {refexp2}',
         "1: {target1} 2: {target2}"],
        ['State {property1} and {property2} of the image bounding boxes that match {refexp1} and {refexp2}',
         "{target1} and {target2}"],
        ['What is {property1} of the image region {refexp1} refers to? What is {property2} of the region {refexp2} refers to?',
        "{target1}. {target2}."],
        ['Expression1: {refexp1} Expression2: {refexp2} Property1: {property1} Property2: {property2} '
         'Instruction: Return property 1 of the region matching expression 1, and property 2 for the region matching expression 2.',
         "{target1}. {target2}."],
        ['If there is a rectangle around "{refexp1}" what is {property1}? Also give me the {property2} for "{refexp2}"?',
         "{target1}. {target2}."]
    ]
)


PROMPT_DICT['Refexp Augmentented Select'] = dict(
    original=[
        ['Which of the these image regions is {comparison}, 1: {refexp1} or 2: {refexp2}', "1", "2"]
    ],
    manual=[
        ['Expression 1: {refexp1} Expression 2: {refexp2} '
         'Instruction: Which expression is {comparison}? ', "1", "2"],
        ['If there was a bounding box around \"{refexp1}\" and \"{refexp2}\" '
         'would the first box be the {comparison}', "yes", "no"],
        ['If the image region corresponding to \"{refexp1}\" {comparison} compared to '
         'the image region corresponding to \"{refexp2}\"?', "yes", "no"],
        ['Is the {refexp1} {comparison} relative to {refexp2}?', "yes", "no"]
    ]
)


PROMPT_DICT['Joint Property Refexp Augmentented'] = dict(
    original=[
        'Given the image region matching {refexp1} and the one matching {refexp2}, return {property}',
    ],
    manual=[
        'Region 1: {refexp1} Region 2: {refexp2} Instruction: For these image regions, return {property}',
        "Tell me {property} for the image regions \"{refexp1}\" and \"{refexp2}\"",
        "In this image there is a {refexp1} and a {refexp2}, what is the {property}",
        "If there were bounding boxes in this image around {refexp1} and {refexp2}, tell me the {property}?"
    ]
)


PROMPT_DICT['llava_instruct_multiple']['manual'] = [
    '{}',
    'Answer the following questions:\n{}',
    'Questions:\n{}', "{}\nAnswer those questions one by one.",
    "Answer questions in a paragraph: {}",
]


PROMPT_DICT['Image_Tagging_Scene'] = {
    "original": [
        'Name the scene shown in this image.',
    ],
    "manual": [
        'Give me a very short description of the scene of <image_input>.',
        'Describe this scene in only a couple of words',
        'Describe the setting in a few words',
        'Name the setting of this picture',
        'Name the setting of <image_input>',
        "What could you categorize this scene as?",
        'Where is the image taking place? Answer in only a few words',
        'What could this scene be categorized as?',
        'Name the high-level semantic class of the situation shown here.',
        'Tell me very briefly where <image_input> is set.',
        'I want to categorize this scene. Tell me a high-level category that matches it.',
        'What would a high-level tag of this image\'s setting be?',
        'How would you summarize this picture\'s setting in a word or two?',
    ]
}

Image_Tagging_Fine_Grained = [
    'What is the most prominent object in this image? Please provide a short answer.',
    'Name the most prominent object in this image with a short answer.',
    'Give me the name of the object in the image',
    'Give me a fine-grained tag for the thing in the photo',
    'Generate a fine-grained tag for the thing in <image_input>',

    'Please provide a short answer. What does <image_input> contain?',
    "What is this? Give me a precise but very short, few-word answer.",
    "What is the main object in the photo? Return a short but fine-grained category.",
    "Return a fine-grained class that matches the object in the image.",
    "State, precisely and succinctly, what is shown here",
    "Tell me a fine-grained category that matches the object in the image.",

    'Can you identify the object in <image_input>? Give a brief response.',
    'Identify the object in the image. Brief response:',
    'Can you describe briefly what\'s in the image?',
    'What does the image show? Kindly give a concise answer.',
    'In a few words, can you tell what object is depicted in the image?',
    'Could you briefly explain what object is presented in the image?',

    'Identify the main object in this image in a few words',
    'Provide a brief description of the most noticeable object in this image.',
    'What\'s the single object highlighted in this image?',
    'Briefly, what is the primary content of this image?',
    'What object stands out the most in this image? Please give a succinct response.',
    'Name the object that is the most striking in this image briefly.',

    'What is the principal item featured in this image? Keep the answer short.',
    'Offer a concise response: What\'s the main focus of this image?',
    'Please provide a quick answer. What\'s the chief subject of the image?',
    'Can you identify the central subject in this image? A short answer will do.',
    'What\'s the foremost entity captured in this picture? Keep your response brief.',
    'Give a succinct description of the primary thing in this picture.',
    'In this image, what\'s the dominant figure or object? Please provide a short answer.',
    'What\'s the major object in this image? Kindly provide a succinct reply.',
    'Quickly name the primary object featured in this picture.',
    'Categorize the key entity depicted in the image in a few words.',
    'Which object stands out the most? Give a short answer.',
    'In just a few words, what\'s the notable object in this image?',
]

PROMPT_DICT['image_tagging_imagenet2012'] = {
    "original": ["What is this? Return a precise but very short answer."],
    "manual": Image_Tagging_Fine_Grained
}

PROMPT_DICT['image_tagging_i_naturalist2017'] = {
    "original": ["What is the scientific name of this animal?"],
    "manual": [
        "Determine the scientific name of this animal.",
        "Tell me the scientific name of the animal in <image_input>",
        "State the scientific name of this species.",
        "What is this? Answer with a scientific name.",
        "Help me figure out the scientific name of the animal in the image.",
        "This image contains an animal, what is its scientific name?",
        "Return the scientific name of the animal shown in the photo.",
        "I want to learn the scientific name of an animal. A picture of the animal is here. What is the name?",
        "What would a biologist call this animal?",
        "What is the scientific category of this animal?",
        "How would a biology textbook identify this animal?",
        "Give me a scientific category of the thing in this photograph.",
        "What is the Latin name of this animal?",
        "What is the binomial name of this creature?",
        "What is the binomial name of the creature in <image_input>?",
        "This photograph contain an animal, what is it's scientific name?"
    ]
}
PROMPT_DICT['image_tagging_caltech_birds2011'] = {
  "original": ["What kind of bird is this?"],
  "manual": [
    "What animal is this? Be specific.",
    "Name the species of this creature.",
    "State the species of this creature in <image_input>.",
    "What type of bird is this?",
    "Identify the animal in the photograph.",
    "Identify this bird",
    "I am not sure what type of bird is in the picture, what do you think?",
    "Help me figure out what kind of bird this is.",
    "What type of bird is in <image_input>?",
    'I saw this bird when bird watching, what kind of bird is it?',
    "This is an image of a bird, what is the exact species?",
   ] + Image_Tagging_Fine_Grained
}

# box classification:

PROMPT_DICT['Box_Classification_Scene'] = {
    "original": ['What is the category of the object at {}?'],
    "manual": [
        'What is the category item in the region {}?',
        'What is the category of the object in the region " {} "?',
        'What kind of object is in the region {} of <image_input>?',
        'What type object is located at " {} "?',
        'State the type object that is located in this bounding box: {}.',
        'There is an object at {}, please categorize it.',
        'Categorize {}',
        'Tag {} in <image_input>',
        'Tag the object at {}.',
        'Instruction: Categorize the object\nContext: {}\n',
        'Name the object in {}',
        'Name the object in {} of the photograph',
        'Help me figure out what kind of object is located at {} in <image_input>',
        'Region: {}\nInstruction: Name the object in the region\n',
        'Very briefly, tell me what object is in " {} ".',
    ],
    "gpt3": [
        'Can you identify the type of object that is located at {}?',
        'Name the object in the region: {}',
        'What type of object is present in the area " {} "?',
        'What is the object situated in the " {} " region? Respond with just a few words.',
        'Which object is positioned at " {} "?',
        'An object exists at {}, can you identify it?',
        'What category of item is located in the zone " {} "?',
        'What is the item in the " {} " area?',
        'There\'s something at {}, can you tell what it is?',
        'What kind of object can be found in {}?',
        'Identify the object in the region {}.',
        'Can you specify the object\'s category in the area {}?',
        'Which object is present in the " {} " region?',
        'An object is spotted at {}, can you specify it?',
        'What class of object lies in the part " {} "?',
        'What is the object in " {} " area?',
        'Which object resides in " {} "?',
        'Something is located at {}, what would you categorize it as?',

        'What kind of object can be found in the section {}?',
        'Identify the object in the region {}.',
        'Can you specify the object\'s category in the area {}?',
        'What class of object lies in the part {}?',
        'Briefly describe the object in {} area?',
        'Give a few word description of the object that resides in {}?',
    ]
}

# image caption
PROMPT_DICT['image_caption_coco_2017'] = {
    "original": ['Caption this image.'],
    "manual": [
        'Write a caption',
        'Write a caption for <image_input>',
        'Please create a caption',
        'Write a caption for this picture.',
        'Briefly describe this picture.',
        'Can you write a caption for this image?',

        'Caption this image with short description.',
        'Caption <image_input> with one or two sentences.',
        'Generate a caption describing this scene',
        'Provide a one to two sentence high-level summary of this photograph.',
        'What is this picture of? Answer with a short sentence.',
        'Write a short caption of <image_input>',
        'Write a short description of this image.',
        'Write a brief caption describing the image.',
        'A short image caption:',
        'Write a short description for the image.',
        'Briefly describe <image_input>.',
        'Concisely describe this image.',
        'Brief summary of the image:',
    ]
}

PROMPT_DICT['Image_Captioning_TextCaps']['manual'] = [
    "Provide a one-sentence caption for the provided image.",    # llava 1.5
    "Examine the text in the image and describe its visual and textual content in a single caption.",
    "Generate a caption that captures both the visual elements and the textual content of the image.",
    "Read and interpret the text within the image and provide a comprehensive short caption.",
    "After analyzing the text and visual content in the image, describe it in one concise caption.",
    "Consider the text present in the <image_input>, and craft a descriptive one-sentence caption.",
    "Incorporate insights from the text within the image and generate a descriptive single-sentence caption.",
    "Detail the image by describing both its visual scene and the text it contains. One-sentence caption:",
    "After closely reading the text in the image, generate a caption for <image_input>.",
]

PROMPT_DICT['image_caption_open_image_v6'] = {
    "original": ['Describe the different things you see when looking at this image.'],
    "manual": [
        'Detail the different things that are in this scene.',
        'Create a visual narrative for this scene by describing the various individual elements inside it.',
        'What do you see in this image? Describe the individual element you observe.',
        'Tell me about the various things you can see in the image.',
        'Generate text about the various things you can see in <image_input>.',
        'What different things can be seen in this photograph?',
        'Tell me what different things can be viewed in this <image_input>',
        'Explain what is in the image. Focus on describing individual parts of the image rather than giving an overall summary.',
        'Describe the different individual components of this photograph.'
    ],
}

PROMPT_DICT['image_caption_vizwiz'] = {
    "original": ['What was this person taking a picture of? Respond with a short phrase.'],
    "manual": [
        'Briefly describe the thing that is the main focus of the image.',
        'What did the photographer take a picture of? Short answer: ',
        'What am I looking at? Short answer: ',
        'Answer with few words. Tell me what I took a picture of',
        'This image was taken by someone who is visually impaired, how you would describe it to them? Provide the answer as short as possible.',
        'A visually impaired person took this picture, can you help them identify the subject of the photograph? Short answer: ',
        'This image was taken by a visually impaired person, caption it in way that would be helpful to them. Short answer: ',
        'This is a photograph from a visually impaired individual, provide a short yet helpful caption.',
        'This image was photographed by a visually impaired individual. Generate a concise but useful description.',
    ],
}

PROMPT_DICT["vqa_vizwiz"]["original"] = [
    # llava1.5
    "When the provided information is insufficient, respond with 'unanswerable'. Answer the question using a single word or phrase: {}",
]

PROMPT_DICT['vqa_vizwiz']["manual"] = [
    "If the image doesn't provide enough details to answer, reply 'unanswerable'. Provide a concise response to the question: {}",
    "Given the <image> taken by phone, answer the question as short as possible: {}",
    "Question: {}\nTask: answer the question with a single word or phrase; and respond with `unanswerable` if the question cannot be answered.",
    "{} A short answer to the question is",
    "Question: {}\nInstruction: Write a short answer to the question using <image_input>\nAnswer:",
]


PROMPT_DICT["vqa_vizwiz_grounding"]["original"] = [
    # llava1.5
    "Task 1: When the provided information is insufficient, respond with 'unanswerable'. Answer the question using a single word or phrase: {}\n"
    "Task 2: Create a segmentation mask highlighting the relevant portions of the image that helps answering Task 1.",
]

PROMPT_DICT['vqa_vizwiz_grounding']["manual"] = [
    "If the image doesn't provide enough details to answer, reply 'unanswerable'. Provide a concise response to the question: {}\n"
    "What is the segmentation mask of the object that related to that question?",
    "Question: {}\nTasks: 1. Answer with the answer as short as possible; 2. Generate a segmentation mask of the object related to the question.",
    "{} When the provided information is insufficient, respond with 'unanswerable'. Answer the question using a single word or phrase, and also return the region in the image used to arrive at the answer.",
]


PROMPT_DICT['Region_Captioning']['manual'] = [
    'Caption the region " {} " of the image with a few words.',
    'Caption the region {} of <image_input> with a few words.',
    "Write a short of {} in <image_input>",
    'Write a short description of the region " {} ".',
    'Please use a few words to describe the region " {} ".',
    'Write a brief caption describing the region " {} ".',
    'Please provide a short answer. Write a caption specifically describing the region " {} ".',
    'Caption the region {} of the image with a few words.',
    'Write a short description of the region {}.',
    'Please use a few words to describe the region {}.',
    'Write a brief caption describing the region {}.',
    'Please provide a short answer. Write a caption specifically describing the region {}.',
]

PROMPT_DICT['Region_Captioning']['gpt3'] = [
    'Could you concisely describe the region " {} "?',
    'Pen a succinct description for the region " {} ".',
    'Offer a short answer. Caption the area identified as " {} ".',
    'Draft a condensed caption for the area " {} ".',
    'Offer a terse response. Describe the section labeled as " {} ".',
    'Caption the " {} " region of the image in few words.',
    'Could you summarise the " {} " region briefly?',
    'Craft a brief caption for the section " {} ".',
    'Please respond briefly. Caption the {} area of the image.',
    'Describe the area {} briefly.',
    'Develop a terse caption for the {} part.',
    'Give the {} region in the picture a concise caption.',
    'Compose a succinct description of the {} region.',
    'Can you give a brief explanation of the {} region?',
    'Jot down a short caption for the {} area.',
    'Offer a succinct reply. Describe the {} region in the picture.',
    'Pen a concise account of the {} section.',
    'Could you briefly describe the {} zone in the image?',
    'Provide a terse response. Caption the identified area {}.',
]



PROMPT_DICT['Prefix']['original'] = [
    'Start your output with {prefix}.'
]
PROMPT_DICT['Prefix']['manual'] = [
    "Prefix the response with {prefix}.",
    "Please begin your output with {prefix}.",
    "Write \"{prefix}\" before your response.",
    "Start by generating \"{prefix}\"."
]

PROMPT_DICT['caption_property']['original'] = [
    "Make sure it {spec}",
    "Ensure it {spec}",
]

PROMPT_DICT['caption_words']['original'] = [
    "contains these words: {words}",
    "has these words: {words}",
    "includes the following words {words}",
]


PROMPT_DICT['Prefixes_Captioning']['original'] = [
    "My caption is:",
]
PROMPT_DICT['Prefixes_Captioning']['manual'] = [
    "Caption:",
    "Text:",
    "Image:",
    "Here is my caption.",
    ">>"
]


IMAGE_GENERATION_NOISY = [
    "Generate an image that roughly matches this text: {}",
    "Build an image related to: \"{}\"",
    "Build an image that could match \"{}\"",
    "Description: {}\nInstruction: Build an image that roughly matches the description",
    "Draw something related to this text: {}",
    "Construct an image that is associated with \"{}\"",
    'What is an image that is associated with the text "{}"?',
    "Text: {}\nPlease build an image related to the text.",
    "{}\nConstruct a related picture.",
    "{}\nCan you draw a related image?",
    "\"{}\" could be associated with what image?",
]

WEB_IMAGE_GENERATION = dict(
    original=["Generate an image that might belong to a webpage that includes this text: {}"],
    manual=IMAGE_GENERATION_NOISY + [
        "Generate an image that might belong to a webpage that contained this text: {}",
        "Imagine a webpage that includes the text \"{}\", draw a related image that might be on that page.",
        "Generate an image that could match the alt-text: {}",
        "For the alt-text: {}, generate an image that could match it.",
    ]
)

PROMPT_DICT['image_generation_cc3m'] = WEB_IMAGE_GENERATION
PROMPT_DICT['image_generation_cc12m'] = WEB_IMAGE_GENERATION
PROMPT_DICT['image_generation_laion_aesthetics_v2_5'] = WEB_IMAGE_GENERATION


PROMPT_DICT['image_generation_redcap'] = dict(
    original=["For a post on Reddit with title {}, what might be a matching image?"],
    manual=IMAGE_GENERATION_NOISY + [
        'Generate an image that Reddit would describe as {}',
        'Draw an image that would be captioned {} on Reddit.',
        'If create a post on an online forum with the title {}, what might the image be?',
        'Come up with an image and would match the title {} on a website'
    ]
)


PROMPT_DICT['image_generation_coco_2017'] = dict(
    original=['Generate an image matching this text: {}'],
    manual=[
        'What image corresponds to the text description "{}"?',
        'Complete the image corresponding to the caption: "{}"',
        'What image can be described as "{}"?',
        'Construct a picture that could have this caption: {}"?',
        'Text: {}\nGenerate an image that matches the text.',
        'Draw an image that can be described as "{}".',
        'Draw an image that could be captioned "{}".',
        'Generate an image that could be captioned "{}".',
        'Caption: {}\nWhat is an image that matches the caption?',
        'What might \"{}\" look like?',
        'Generate an image of {}',
        'Generate a picture of {}',
        'Scene: {}\nInstruction: Please draw the scene',
    ],
    gpt=[
        'Draw an image that matches the caption "{}"',
        'Illustrate an image that aligns with the description "{}".',
        'Create an image that might be captioned as "{}".',
        'Conceive an image that would suit the caption "{}".',
        'Fabricate an image with a possible caption of "{}".',
        'Render an image that matches {}.',
    ]
)

# PROMPT_DICT['image_generation_coco_2014'] = PROMPT_DICT['image_generation_coco_2017']
PROMPT_DICT['image_generation_openimages'] = dict(PROMPT_DICT['image_generation_coco_2017'])

PROMPT_DICT['image_generation_coco_2014']['original'] = [
    "An image of {}"
]

# VQA
PROMPT_DICT['VQA_short_prompt']["manual"] = [
    "{} A short answer to the question is",
    "{} Short answer:",
    "Answer this question very succinctly: {}",
    "Please provide a short answer. {}",
    "Look at <image_input> and then give a brief answer to this question: {}",
    "Question: {}\nInstruction: Write a short answer to the question using <image_input>\nAnswer:",
    "Give a very brief answer. {}",
    "Given the image, answer the following question with no more than three words. {}",
    "Use the provided image to answer the question: {} Keep your answer as short as possible:",
    'The question "{}" can be answered using the image. A short answer is',
    '{} Answer the question as briefly as possible.',
    'The short answer of the question "{}" is:',
    "{} Answer the question with a single word or phrase",
]


PROMPT_DICT['VQA_sentence']["manual"] = [
    "{} A short sentence answer to the question is",
    "Given the image, answer the following question with one sentence. {}",
    "Use <image_input> to answer the following question with one sentence. {}",
    "{} Based on the <image_input>, respond to this question with a short completed sentence:",
    '{} Answer the question with a complete sentence as briefly as possible.',
    'Question: {}\nAnswer it with one sentence.',
]

PROMPT_DICT['TallyQA']["manual"] = [
    "{} Answer the question with the number only.",
    "Question: {}\nAnswer it with the number only without any explanation.",
    "{} Based on the <image_input>, answer this question with a integer:",
    '{} Answer the question as briefly as possible.',
]


PROMPT_DICT['VSR Binary']["manual"] = [
    "Judge whether the relationship is described correctly: {} Answer with `yes` or `no` only.",
    'Is the relation "{}" true? Check the image <image_input> and answer the question with only `yes` or `no`.',
    '<image_input> {} Is this statement true? Answer with "yes" or "no" only.',
]


PROMPT_DICT['VSR Full']["manual"] = [
    "Describe shortly about what is the relation between {#SUBJ} and {#OBJ} in <image_input>",
    "What is the relation between {#SUBJ} and {#OBJ}? Provide a concise sentence for the answer.",
    "What is the spatial relation between {#SUBJ} and {#OBJ} in <image_input>? Answer with the short answer.",
]


PROMPT_DICT['VCR_QA']['original'] = [
    'Question: {1}\nOptions: {2}\nTask: Select the best answer only based on <image_input>.',
    'Task: State the best answer option only.\nQuestion: {1}\nOptions: {2}',
]

PROMPT_DICT['VCR_QA']['manual'] = [
    'Answer the question " {1} " by picking from these options: {2}. Short answer:',
    'Question: {1}\nAnswer options: {2}\nChoose an answer from the given options.',
    'Question: {1}\nAnswer options: {2}\nWhich option answers the question with the most common sense?',
    'Which option answers the question with the most common sense? Question: {1}\nAnswer options: {2}',
    'Answer the common sense question with a few words " {1} " by picking from these options: {2}',
    # 'Use common sense reasoning to concisely answer the question "{1}". The answer options are "{2}"'
]

PROMPT_DICT['VCR_QAR']['original'] = [
    'Question: {1}\nAnswer: {2}\nRationale Options: {3}\nTask: Select the best rationale option.',
]

PROMPT_DICT['VCR_QAR']['revised_original'] = [
    # 'Question: " {1} " Answer: " {2} " Rationale: " {3} ".',
    'Question: {1}\nAnswer: {2}\nRationale Options: {3}\nTask: Select the best rationale option.',
]

PROMPT_DICT['VCR_QAR']['manual'] = [
    'Based on the image, the answer to the question " {1} " is " {2} ". What rationale makes the most sense for this? Options: {3}',
    'Common sense reasoning says that the answer to " {1} " is " {2} ". Why is this the case? Pick from the following options: {3}'
]

PROMPT_DICT['Prompt_ScienceQA_QCM_A']['original'] = [
    'Select the best answer option to the question using the context.\n{}',
    'Task: State the best answer option only.\n{}',
    '{}\nTask: Select the best answer only.',
    '{}\nProvide the short answer.',
    '{}\nAnswer: '
]


PROMPT_DICT['Prompt_ScienceQA_QCM_ALE']['original'] = [
    'Select the best answer option to the question using the context and explain why. \n{}',
    'Task: State the best answer option and explain in details. \n{}',
    '{}\nTask: Select the best answer, and justify your answer.',
]

PROMPT_DICT['SEEDBench_Prompt']['manual'] = [
    'Select the best answer option to the question using the context.\nQuestion: {}',
    'Task: State the best answer option only.\nQuestion: {}',
    'Question: {}\n Task: Select the best answer only.',
    'Question: {}\n Provide the short answer.',
    'Question: {}\nAnswer: '
]

PROMPT_DICT['MMBench_Prompt']['manual'] = [
    # "Answer with the option's letter from the given choices directly.",
    'Select the best answer option to the question using the context.\nQuestion: {}',
    'Task: State the best answer option only.\nQuestion: {}',
    'Question: {}\n Task: Select the best answer only.',
    'Question: {}\n Provide the short answer.',
    'Question: {}\nAnswer: '
]


PROMPT_DICT['Visual_Entailment']['original'] = [
    'Can image and text " {1} " imply text " {2} "? Give me a short answer.'
]

PROMPT_DICT['Visual_Entailment']['revised_original'] = [
    'Can the image and text {1} imply text {2}? Short answer:',
]

PROMPT_DICT['Visual_Entailment']['manual'] = [
    'Based on this image, is the relationship between " {1} " and " {2} " entailment, contradiction, or neutral?',
    'Based on this image, is the relationship between {1} and {2} entailment, contradiction, or neutral?'
]

PROMPT_DICT['Relationship_Tagging']['original'] = [
    'What is the relationship between " {1} " and " {2} " ?'
]

PROMPT_DICT['Relationship_Tagging']['revised_original'] = [
    'What is the relationship between " {1} " and " {2} "?',
]

PROMPT_DICT['Relationship_Tagging']['manual'] = [
    'What relationship does " {1} " have with " {2} "? Short answer:',
    'Use a few words to describe how are " {1} " and " {2} " related.',
    'Please provide a short answer. What is the relationship between the two objects defined by the location and class " {1} " and " {2} "?',
    'Can you briefly describe what relationship does " {1} " have with " {2} ?',
    'Provide a short account of the relationship between " {1} " and " {2} ".'
]

PROMPT_DICT['Relationship_Tagging_Aug']["manual"] = [
    'Enumerate all related objects relevant to " {1} ".',
    'Output all objects and relationships of " {1} ".'
]

PROMPT_DICT['Prefix LM'] = dict(
    original=[
        "Continue this text: ",
    ],
    manual=[
        "Extend this passage: ",
        "Guess what might come next: ",
        "Generate some coherent text that might be found in a book or webpage. The text starts with: ",
        "Instruction: Write a plausible continuation of this passage\nPassage: ",
        "Write a plausible continuation: ",
        "What might follow this text? ",
    ]
)


PROMPT_DICT['Det Augmentented'] = dict(
    original=[
        'Output {property} and category of the regions in the image described by: {refexp}'
    ],
    manual=[
        'What is {property} and category of the regions {refexp} refers to?',
        'If there was a rectangle around {refexp}, what would {property} and category be?',
        'Expression: {refexp} Instruction: What is {property} and category of the regions the expression refers to?',
        'For the regions in the image that matches the expression "{refexp}", return {property} and category',
        'In the image there are regions containing: {refexp}, return {property} and category',
        'Instruction: Compute {property} and the category of the image regions that matches the description. Description: {refexp}',
        'Figure out what bounding boxes in the images match {refexp}, and then return {property} and category',
    ],
    gpt3=[],
)


PROMPT_DICT['Refexp'] = {
    "original": ['Which region does the text "{}" describe?'],
    "manual": [
        "Expression: {}\nInstruction: Return the region in the image that matches the expression",
        "What region does \"{}\" match?",
        "What region would a person describe as: {}",
        "Generate a bounding box around {}",
        "Find a region in <image_input> that contains {}",
        "Find me the image region that contains {}",
        "Show me the region in <image_input> that best matches \"{}\"",
        "What region in the picture contains {}?",
        "Report a bounding box that encompasses {}",
        "For the expression \'{}\', what image region matches it?",
        "Identify the image region that best matches the expression: {}",
        "Help me find a particular region in the image.\nDescription: {}\nRegion:",
        "I want to find the {}, can you tell me what region it is in?",
    ]
}


PROMPT_DICT['Object_Detection'] = {
    "original": ['Return the bounding boxes and categories of region matching "{}"'],
    "manual": [
        'Which regions can be described as "{}"? Return their bounding boxes and object categories.',
        'Return the bounding boxes and category names of instances of \"{}\" in the image',
        'Find instances of {} in the <image_input> and return their bounding boxes and category names',
        'Categories: {}\nInstruction: Find instances of the given categories in the image and return their bounding boxes and category names',
        'Find regions containing \"{}\", label each region with its category.',
        'Find the {}, label each region with its category.',
        'Report the region and categories of {} in the image.',
        'Find instances of \"{}\", return the y1 x1 y2 x2 coordinates and category of each instance.',
    ],
}


PROMPT_DICT['Object_Detection_No_Cat'] = {
    "original": ['Which regions does the text "{}" describe?'],
    "manual": [
        "Objects: {}\nInstruction: Return the location of the objects in the image",
        "I am looking for {} in this picture, can you show me where they are?",
        "What regions contain {} in <image_input>?",
        'Find image regions that match: {}',
        'Find all instances of {}.',
        'Please find all instances of {} in <image_input>',
        'Locate all instances of {}.',
        'List the regions of the image that match {}.',
        'Find all regions in the image corresponding to the category "{}".',
    ],
    "gpt3": [
        'Find every "{}" in this image.',
        'Please locate all of the {} in this image.',
        'Search the image for all regions corresponding to the text "{}".',
        'Identify all "{}" regions in the image.',
    ]
}


PROMPT_DICT['Object_Detection_obj_aug']['manual'] = [
    'Find the instance of {} in {SPECIAL}.',
]


PROMPT_DICT['Box Counting']['manual'] = [
    'How many instances of "{}" in this image? Short answer: ',
    'Count the number of "{}" in this image. Answer with the number only.',
    'Can you identify how many "{}" in this image?',
    'See if you can find and count the number of all of the "{}" in this image.',
    'Identify the number of all "{}" in the image.',
    'Discover all "{}" in the image. How many are there? Provide me the number only.',
    'The number of "{}" shown in <image_input> is:',
    'The number of "{}" contained in the image is ',
]

PROMPT_DICT['Box Counting Denoising']['manual'] = [
    'There are {SPECIAL} "{}" in the image.',
    'Question: How many {} in the image? Answer: {SPECIAL}',
    'The number of "{}" shown in <image_input> is {SPECIAL}.',
    'The number of "{}" contained in the image is {SPECIAL}.',
    'The number of all "{}" in the image is {SPECIAL}.'
]

# Unused
# PROMPT_DICT['Object_Detection_Modality_Token']['manual'] = [
#     'Which region does the text "{}" describe in <image_input>?',
#     'Which region can be described as "{}" in <image_input>?',
#     'Find the occurrences of {} in this <image_input>.',
#     '<image_input> Find all instances of {}.',
#     '<image_input> Locate all instances of {}.',
#     'Locate all instances of {}. <image_input>',
#     'Find all regions in the <image_input> corresponding to the category "{}".',
#     'Look for all instances of "{}" in this <image_input>.',
#     'Find every "{}" in this <image_input>.',
#     'Can you spot all the "{}" in this <image_input>?',
#     'See if you can locate all of the "{}" in this <image_input>.',
#     'Search the <image_input> for all regions corresponding to the caption "{}".',
#     'Identify all "{}" in the <image_input>.',
#     'Discover all "{}" in the <image_input>.',
#     'Spot all of the "{}" in this <image_input>.',
#     'Look for "{}". <image_input>',
#     'There are one or more "{}" in this <image_input>. Can you find them all?',
# ]
# PROMPT_DICT['Object_Detection_All_Class_Modality_Token']['manual'] = [
#     'Locate all objects in the <image_input>.',
#     'Detect all of the objects in the <image_input>.',
#     'Find regions of all of the objects in the <image_input>.',
#     'There are one or more objects in this <image_input>. Find the regions corresponding to each of them.',
# ]
#
# Disabled to use dataset-spefic prompts
# PROMPT_DICT['Object_Detection_All_Class']['original'] = [
#     'Locate all objects in the image.'
# ]
#
# PROMPT_DICT['Object_Detection_All_Class']['revised_original'] = [
#     'Locate all objects in the image.',
# ]
#
# PROMPT_DICT['Object_Detection_All_Class']['manual'] = [
#     'Detect all of the objects in the image.',
#     'Find regions of all of the objects in the image.',
#     'There are one or more objects in this image. Find the regions corresponding to each of them.',
# ]

PROMPT_DICT['Object_Detection_Single_Object']['original'] = [
    'Locate a single object in the image.'
]

PROMPT_DICT['Object_Detection_Single_Object']['revised_original'] = [
    'Locate a single object in the image.',
]

PROMPT_DICT['Object_Detection_Single_Object']['manual'] = [
    'Detect a single object in the image.',
    'Find a region of an object in the image.',
    'There is an object in this image. Find the region corresponding to it.',
]

# Expclity state COCO objects
PROMPT_DICT["detection_coco_2017"] = {
    "original": ["Find the COCO objects."],
    "manual": [
        'List the COCO objects and their locations in the image',
        'Locate all COCO objects in the image.',
        'Detect all of the COCO objects in the image.',
        'Find all the COCO objects.',
        'Find regions of all of the COCO objects in the image.',
        'There are one or more COCO objects in this image. Find the regions corresponding to each of them.',
        'For COCO object in the image, return the object category and bounding box'
    ]
}

Object_Detection_All_Generic = [
    'Enumerate the objects in the picture. For each one, list its category and bounding box.',
    'Locate all objects in the image.',
    'Detect all of the objects in the image.',
    'Detect all of the objects in <image_input>.',
    'Find all the objects.',
    'Find regions of all of the objects in the image.',
    'There are one or more objects in this image. Find the regions corresponding to each of them.',
    'For each object in the image, return the object category and bounding box',
    "Perform object detection by returning a class and bounding box for each object in this image"
]


PROMPT_DICT["detection_open_images_v6"] = {
    "original": ['List all objects and their locations in the image'],
    "manual": Object_Detection_All_Generic
}
PROMPT_DICT["detection_vg"] = {
    "original": ['List all objects and their locations in the image'],
    "manual": Object_Detection_All_Generic
}

PROMPT_DICT["video_single_object_detection_youtube_bb"] = {
    "original": ['List all objects and their locations in the image'],
    "manual": Object_Detection_All_Generic
}

PROMPT_DICT['OCR Detect All']['manual'] = [
    'Detect all visible alphabetic or numerical texts in the image. Answer with the words only.',
    'Identify and extract all texts present in the image. Answer with the text only.',
    "Identify and extract all visible texts present in the image. Answer with texts separated by ','.",
    "Highlight any readable alphabetic or numerical text within the photo. Output the text that you see only.",
    "Locate and transcribe the textual content from the picture. Visible texts:",
    "Find and list all alphabetic or numerical phrases displayed in the image. Predict with the text you saw only.",
    'Now you are an OCR agent. Now display all visible characters in the image only.',
    'Imagine you are an OCR agent. Display all characters you saw in the image.',
    'Extract the alphabetic or numerical texts presented in the image:',
    "As an OCR tool, retrieve and list all visible text in the image. Answer with text only.",
    "Without adding extra words, transcribe any visible text from the image.",
    "Analyze and return the legible content from the photo. Output text only.",
    "Your task is to identify words or numbers in the picture. Provide them separated by ','.",
    "Operate as an OCR scanner and list all the visible text you find in the image.",
    "You are an Optical Character Recognition (OCR) agent and you will help me recognize all alphabetic or numerical characters in the image. Now process the given image and return the texts:",
]


PROMPT_DICT['OCR Detect All and bbox']['manual'] = [
    'Detect all visible alphabetic or numerical texts with their bounding boxes in the image.',
    'Identify and extract all texts present in the image. Answer the texts and corresponding bounding box in the format of y1x1y2x2.',
    "Identify and extract all regions containing visible texts presented in the image. Answer with bounding boxes and their texts.",
    "Operate as an OCR scanner and list all the visible text as well as their corresponding bounding boxes you find in the image.",
    "Task: Optical Character Recognition (OCR)\nContext: <image_input>\nOutput: bounding box coordinates with legible characters inside.",
]


PROMPT_DICT['OCR Detect All bbox input']['manual'] = [
    'Detect all visible alphabetic or numerical texts with bounding boxes " {} " in the image. Answer the texts only.',
    'In region " {} ", Identify and extract all texts present in the image. Answer the texts only.',
    "Operate as an OCR scanner and list all the visible text you find in the image. Specific region: {}",
    "You are an Optical Character Recognition (OCR) agent and you will help me recognize all alphabetic or numerical characters in the image. Now process the given image and return the texts within geven region: {}.",
    "Task: Optical Character Recognition (OCR)\nContext: y1x1y2x2 bounding boxes ' {} ' in <image_input>\nOutput: legible texts.",
]


PROMPT_DICT['OCR Detect All bbox input include negative']['manual'] = [
    'Detect all visible alphabetic or numerical texts with bounding boxes " {} " in the image. If no text legible in any of the region, return "not legible". Answer the texts only.',
    'In region " {} ", Identify and extract all texts present in the image (say "not legible" if any region does not contain characters). Answer the texts only.',
    "Task: Optical Character Recognition (OCR)\nContext: y1x1y2x2 bounding boxes ' {} ' in <image_input>\nOutput: legible texts and 'not legible' if illegible",
]


PROMPT_DICT['OCR Detection Task']['manual'] = [
    'Locate the region of the characters: "{LABELS}".',
    "What is the bounding box of the legible texts '{LABELS}'?",
    "Where can the characters '{LABELS}' be found on the screen? Output in bounding box format.",
    "Mark the section displaying the characters '{LABELS}'.",
    "Task: Locate all characters in the image\nTexts: {LABELS}\nOutput format: y1x1y2x2 bounding box",
    'In which area does the characters "{LABELS}" appear? Output bounding box in the format of y1x1y2x2.'
]


PROMPT_DICT['Object_Segmentation']['original'] = [
    'Segment the object in this bounding box: {}',
]

PROMPT_DICT['Object_Segmentation']['manual'] = [
    'Building a binary mask of the pixels that part of the main object in {}',
    'Generate a black and white image. The white pixels should be ones that are part of the object in {}',
    'Segment the object at {}',
    'There is an object at {}. Building a binary mask containing just the pixels of that object',
    "Object segmentation requires building a binary mask with white pixels that are part of a particular object and black pixels"
    " everywhere else. Do object segmentation for the object at {}",
    "Show me what pixels are part of the object at {}",
    "What are the exact pixels that belong to the main object in the bounding box {}?"
]

PROMPT_DICT['Object_Segmentation']['gpt3'] = [
    'Select the pixels that match the description " {} ".',
    'Highlight the pixels that fit the description " {} ".',
    'To do object segmentation, you must find the pixels that represent each instance of a category. Find the object segmentation of " {} ".',
    'Identify the pixels that correspond to each instance of the category " {} ".',
    'Mark all the pixels that depict " {} ".',
]

PROMPT_DICT['Multiple Caption Aug S']['manual'] = [
    "{} An image of", "The frame {} in these images shows",
    "{} is an image of",
    "Here are few frames. Describe {} only.",
]

PROMPT_DICT['Multiple Caption Aug Hard']['manual'] = [
    "Describe these frames one by one.",
    "Each frame shows differently. Describe each one:",
]


PROMPT_DICT['VQA_Pointing']['original'] = [
    'Answer this question with a bounding box: {question}'
]

PROMPT_DICT['VQA_Pointing']['manual'] = [
    '{question} Return a bounding box as output.',
    'What region does the answer to this question: \"{question}\" refer to?',
    '{question} Give me the y1 x1 y2 x2 coordinates.',
]

PROMPT_DICT['Region_Class_Segmentation'] = dict(
    original=[
        "Segment {region} for classes {class_name}"
    ],
    manual=[
        "Only for {region}, segment the {class_name}",
        "Region: {region}\nClasses: {class_name}\nSegment the classes in this region",
        "Build a segmentation map of {class_name}, but only for {region}",
        "Show pixels that are part of {class_name} and in {region}",
        "Draw a binary image with white pixels for {class_name} in {region} and black everywhere else",
        "Segment {class_name} that are in {region}",
    ]
)


PROMPT_DICT['Inverse_Class_Segmentation'] = dict(
    original=[
        'Build a binary of mask of all pixels that are not part of {class_name}'
    ],
    manual=[
        'Build an inverted segmentation map for {class_name}',
        "Segment parts of the image that are not part of {class_name}",
        "Segment everything but {class_name}",
        'Show me pixels that are not in these classes: {class_name}',
        'For {class_name}, build binary mask where pixels belonging to the classes are black and other pixels are white',
        'Invert the segmentation for {class_name}',
        "Construct an image where pixels belonging to {class_name} are black and other pixels are white."
    ]
)


PROMPT_DICT['Class_Segmentation']['original'] = [
    'Segment all instances of {class_name}'
]

PROMPT_DICT['Class_Segmentation']['manual'] = [
    'Build a binary mask of pixels that are part of one of these classes: {class_name}',
    'Show me pixels that are part of {class_name}',
    'Generate an image where pixels that are part of {class_name} are white and others are black.',
    'What pixels are part of {class_name}?',
    'For the classes \"{class_name}\" build a binary mask showing pixels that are in that class.',
    "Classes: {class_name}\nInstruction: Build a segmentation for those classes.",
    'Segment the {class_name}',
    'Segment {class_name}.',
]

PROMPT_DICT['Depth_Estimation']['original'] = [
    'What is the depth map of the image ?'
]

PROMPT_DICT['Depth_Estimation']['revised_original'] = [
    'What is the depth map of the image ?',
]

PROMPT_DICT['Depth_Estimation']['manual'] = [
    'Assign each pixel in this image a depth value, and generate a grayscale image representing the depth at each pixel.',
    'Determine the depth of each pixel of this image, and generate a map of the same.',
    'Generate a depth map of this image, where darker pixels correspond to less depth.',
    'Depth image: ',
    'Generate the depth image.',
    "What is the depth?",
]

PROMPT_DICT['Depth_Estimation']['gpt3'] = [
    "Allocate a depth value to every pixel in the picture and create a grayscale image representing the depth at every pixel.",
    "Ascertain the depth of each pixel in this image and create a corresponding map.",
    "Produce a depth map of the picture where darker pixels signify less depth.",
    "Image demonstrating depth: ",
    "Create an image showing depth.",
    "Could you determine the depth?",
    "Assign a depth value for each pixel in this image and represent it through a grayscale image.",
    "Map the depth of each pixel in this image and generate a corresponding visual.",
    "Form a depth map where the image's darker pixels indicate shallower depth.",
    "Depth portrayal in the image: ",
    "Develop an image that shows the depth.",
    "What's the depth measurement?",
    "Give each pixel a depth value in this image, and form a grayscale image showing the depth of each pixel.",
    "Identify the depth for every pixel in this image and construct a matching map.",
    "Create a depth map for this image, with darker pixels indicating less depth.",
    "Depiction of depth in the image: ",
    "Generate an image that indicates the depth.",
    "Can you provide the depth?",
    "Set a depth value for all pixels in this image and construct a grayscale image to reflect this depth.",
    "Gauge the depth of each pixel in this image and produce a map to show it.",
]


PROMPT_DICT['Surface_Normals_Estimation']['original'] = [
    'What is the surface normal of the image ?'
]

PROMPT_DICT['Surface_Normals_Estimation']['revised_original'] = [
    'What are the surface normals of this image ?',
]

PROMPT_DICT['Surface_Normals_Estimation']['manual'] = [
    'Generate an image depicting the surface normal at each pixel of the input image.',
    'Generate an image representing the orientation of the surface at each pixel.'
]

PROMPT_DICT['Surface_Normals_Estimation']['gpt3'] = [
    'Create an image that shows the orientation of the surface at each pixel.',
    'Generate an image that indicates the orientation of the surface at each pixel.',
    'Make an image that displays the orientation of the surface at every pixel.',
    'Generate the surface normals of this image.',

    "Could you determine the surface normal of the picture?",
    "Produce the surface normals for this image.",
    "Find the surface normals for this particular image.",
    "Can you ascertain the surface normal for this image?",
    "What would be the surface normal for this picture?",
    "Determine the surface normal of the image.",
    "Could you establish the surface normals for this picture?",
    "What's the image's surface normal?",
    "Make the surface normals for this image.",
    "Could you tell me the surface normal for this image?",
    "Formulate the surface normals for this image.",
    "Can you identify the surface normal in the image?",
    "Compute the surface normals for this image.",
    "What is the normal vector for the image surface?",
    "Construct the surface normals for this given image.",
    "What's the normal direction for the image surface?",
]


PROMPT_DICT['Image_Inpainting']['original'] = [
    'Filling the blank region " {} " .',
    'Filling the blank region {}.'
]

PROMPT_DICT['Image_Inpainting']['revised_original'] = [
    'Fill in the blank region " {} ".',
]

PROMPT_DICT['Image_Inpainting']['manual'] = [
    'Complete the image by filling in the region " {} ".',
    'This image may be incomplete. Fill it in with what should go in " {} ".',
    'Complete the image by filling in the region {}.',
    'This image may be incomplete. Fill it in with what should go in {}.'
]

PROMPT_DICT['Image_Inpainting']['gpt3'] = [
    "Fill the vacant area \" {} \" in the image.",
    "Supply content for the blank space \" {} \" in the image.",
    "Complete the image by populating the space \" {} \".",
    "Conclude the image by filling the area \" {} \".",
    "In the image, populate the blank segment \" {} \".",
    "Finalize the picture by filling up the void at \" {} \".",
    "Put content into the vacant slot \" {} \" to finish the image.",
    "Close the gap in the image by filling in \" {} \".",
    "Fill the void at {} to complete the picture.",
    "Plug the hole at {} to finalize the picture.",
    "Complete the visual by filling in the gap at {}.",
    "Make the image whole by filling in {}.",
]


PROMPT_DICT['Pose_Estimation']['original'] = [
    'Find the visible human joints in the region " {} ".'
]

PROMPT_DICT['Pose_Estimation']['manual'] = [
    'Find the visible keypoints corresponding to the person located at {}',
    'Return the visible keypoints on the human body of the person in the region {}',
    'Find the visible human joints of the person located at {}.',
]

PROMPT_DICT['Pose_Estimation']['gpt3'] = [
    'Locate the visible human joints in the region {}.',
    'Where are the visible human joints in the region {} ?',
    'Could you please find the visible joints of a human in the region {}.',
    'Locate the joints of the person located at {}.',
    'Identify the visible joints of the human located at {}.',
    'Find the visible keypoints on the human body in the region {}.',
    'Identify the visible keypoints on the human body appearing at {}.',
    "Identify the noticeable keypoints for the person found at {}.",
    "Spot the visible human joints of the individual positioned at {}.",
    "Detect the discernible keypoints associated with the person at location {}.",
    "Locate the observable keypoints for the person situated at {}.",
    "Point out the visible joints on the human body located at {}.",
    "Track the observable joints of the person in the region {}.",
    "Spot the discernible human joints located in the region {}.",
    "Identify the apparent keypoints for the individual within the region {}.",
    "Find the keypoints for the person stationed at {}.",
    "Locate the discernible joints of the person present at {}.",
    "Detect the human joints in the {} region that are visible.",
    "Track the person's visible keypoints found in the region {}.",
    "Trace the visible keypoints of the person located in {}.",
]

PROMPT_DICT['Image_Inpainting_Part']['manual'] = [
    'Complete the image by filling in the region {}.',
    'This image may be incomplete. Fill it in with what should go in {}.',
    'Build a version of this image with the region {} filled-in.',
    "What would this image look like with {} filled in?"
]

PROMPT_DICT['Image_Inpainting_Part']['gpt3'] = [
    "Fill the region {} in the image to complete it.",
    "Finish the image by populating the region {}.",
    "Complete the image by occupying region {}.",
    "Generate a version of this image where region {} is filled.",
    "In the image, complete the region {}.",
    "Provide a complete picture by filling in region {}.",
    "How would the image appear with region {} filled?",
    "Complete the image's {} region.",
    "Produce a version of this image with {} region filled.",
    "How would this picture look with region {} populated?",
    "Imagine the image with {} filled in.",
    "Complete the picture by filling in {}.",
    "What is the complete image when {} is filled?",
    "Fill in {} to complete the image.",
    "Suppose the image was complete with {} filled.",
    "The image needs completing. Fill in {}.",
    "What would it look like if {} were filled in the image?",
    "Complete the image, make sure to fill in {}.",
    "Imagine and fill the region {} to complete the image.",
    "Conceive the complete image by filling the {}.",
]

PROMPT_DICT['Pose_Estimation_Part']['original'] = [
    'Find the keypoints of {part} in the region {}.'
]

PROMPT_DICT['Pose_Estimation_Part']['manual'] = [
    'Find the keypoints of {part} corresponding to the person located at {}',
    'Return the keypoints of {part} on the human body of the person in the region {}',
]

PROMPT_DICT['Pose_Estimation_Part']['gpt3'] = [
    'Locate the human joints of {part} in the region {}.',
    'Where are the human joints of {part} in the region {} ?',
    'Find the keypoints of {part} on the human body in the region {}.',
    'Identify the keypoints of {part} on the human body appearing at {}.',

    "Pinpoint the {part} joints in the region {}.",
    "Identify the location of {part} joints in the region {}.",
    "Where can the {part} joints be found in the region {}?",
    "Detect the {part} body keypoints in the region {}.",
    "Spot the {part} joints within the region {}.",
    "Identify the position of {part} joints in region {}.",
    "Find the location of the {part} joints in the region {}.",
    "Where are the keypoints for the {part} in the region {}?",
    "Identify the location of {part}'s body keypoints in the {} region.",
    "Can you pinpoint the {part} keypoints in region {}?",
    "Indicate the location of {part}'s joints in the {} region.",
    "Detect the location of the {part}'s joints in region {}.",
    "Spot the location of {part}'s keypoints in the region {}.",
    "Where can you find the {part}'s joints in the region {}?",
    "Locate the {part}'s keypoints within the {} region.",
    "Indicate the position of {part}'s body keypoints in the {} region.",
    "Spot the {part}'s joints within region {}.",
    "Where are the joints of {part} located within region {}?",
    "Can you locate the {part}'s keypoints within the {} region?",
]

PROMPT_DICT['Refer_Expression']['original'] = [
    'Which region does the text " {} " describe ?'
]

PROMPT_DICT['Refer_Expression']['revised_original'] = [
    'Which region does the text " {} " describe?',
]

PROMPT_DICT['Refer_Expression']['manual'] = [
    'Locate the region described by " {} "',
    'Where is the " {} "?',
    'Locate the " {} "',
    'At which region is " {} " present?'
]

PROMPT_DICT['VisComet_Before']['original'] = [
    'Event: " {1} {2} " Before, what the person needed to do ?'
]

PROMPT_DICT['VisComet_Before']['revised_original'] = [
    'Event: "{1} {2}". What did the person need to do before this?'
]

PROMPT_DICT['VisComet_Before']['manual'] = [
    'Event: " {1}". Place: "{2}". What did the person need to do before this?',
    '{1}. This took place {2}. What did the person need to do beforehand?',
    '{1}. This took place {2}. Before this, the person needed to',
    '{2}, {1}. Before this, the person needed to'
]


PROMPT_DICT['VisComet_Intent']['original'] = [
    'Event: " {1} {2} " Because, what the person wanted to do ?'
]

PROMPT_DICT['VisComet_Intent']['revised_original'] = [
    'Event: "{1} {2}", because the person wanted to',
]

PROMPT_DICT['VisComet_Intent']['manual'] = [
    'Event: " {1}". Place: "{2}". What was the person\'s intention?',
    '{1}. This took place {2}. What did the person intend?',
    '{1}. This took place {2}. What did the person want to do?',
    '{1}. This took place {2}. This happened because the person wanted to',
    '{2}, {1}. This happened because the person wanted to'
]

PROMPT_DICT['VisComet_After']['original'] = [
    'Event: " {1} {2} " After, what the person will most likely do ?'
]

PROMPT_DICT['VisComet_After']['revised_original'] = [
    'Event: "{1} {2}". What is the person most likely to do after this?'
]

PROMPT_DICT['VisComet_After']['manual'] = [
    'Event: " {1}". Place: "{2}". What will the person likely do after this?',
    '{1}. This took place {2}. What will the person likely do afterward?',
    '{1}. This took place {2}. After this, the person will likely',
    '{2}, {1}. After this, the person will likely'
]


PROMPT_DICT['Ground_Situation_Recognition_Verb']['original'] = [
    'What is the salient activity of the image ?'
]

PROMPT_DICT['Ground_Situation_Recognition_Verb']['revised_original'] = [
    'What is the salient activity of the image?',
]

PROMPT_DICT['Ground_Situation_Recognition_Verb']['manual'] = [
    'What are they doing?',
    'What are they doing in this image?',
    'What action does this image depict?',
    'What action takes place in this image?'
]


PROMPT_DICT['Ground_Situation_Recognition_Frame']['original'] = [
    'Given the image and salient activity is " {1} ", what is the situation in terms of " {2} " ?'
]

PROMPT_DICT['Ground_Situation_Recognition_Frame']['revised_original'] = [
    'Given the image and salient activity is "{1}", what is the situation in terms of "{2}"?',
]

PROMPT_DICT['Ground_Situation_Recognition_Frame']['manual'] = [
    'This is an image of {1}. What plays each of the following roles in this action? {2}',
    'What are the {2} with respect to the action of {1} in this image?',
]


PROMPT_DICT['Segmentation_based_Image_Generation']['original'] = [
    "This image contains a segmentation map with these colors: {}. Generate an image that matches the segmentation."
]

PROMPT_DICT['Segmentation_based_Image_Generation']['manual'] = [
    'Generate an image based on this segmentation mask. The colors represent these classes: {}',
    'Generate an image matching the semantic segmentation represented by the input image. Each color represents a different class. The mapping is: {}',
    "Create a picture that matches the input segmentation map, the colors represent these classes: {}",
    "Given this color map {}, draw an image that matches the input semantic segmentation.",
    "Generate an image. Try to make it match the input segmentation map, the color mapping is {}",
    "Legend: {}. Instruction: Build an image that matches the input image map using the legend.",
    "Legend: {}. Draw an image matching the given segmentation map with classes matching the ones in the legend."
]

PROMPT_DICT['image_caption_cc3m']['manual'] = [
    'The text corresponding to this image on a website could be',
    'Provide some text that might match the image if it was on a website',
    'Generate some text that could accompany this picture if it was on a webpage.',
    'Write some text that could be from a webpage that references this figure',
    'If this image was on a webpage, what might it\'s alt-text be?',
    'Generate a short statement that is relevant to the image',
    'Generate a short description that could be matched with the picture',
    'Create a brief caption that roughly describes or relates to the image',
    'Write a short sentence that describes or relates to the image',
    'Can you create some text that could match with the image?',
]

PROMPT_DICT['image_caption_cc12m'] = dict(PROMPT_DICT['image_caption_cc3m'])

PROMPT_DICT['image_caption_redcap']['manual'] = [
    'If this photo was posted online, how might it be titled?',
    'How might a webpage containing this image title that image?',
    'If someone posted this photograph online, how might they describe it?',
    'How might someone describe this image if they put it in a blog?',
    'What would someone say about this image if they were uploading it to an online forum?',

    'Create some text that might accompany this image if it was sent to another person',
    'What would a plausible title for the image be if it was posted on a website?',
    'How would this image be titled on Reddit?',
    'If this image was posted on reddit.com, what could be its title?',
    'Make up a title for this image that could be used if it was uploaded to reddit.'
]


PROMPT_DICT['controllable_image_gen']['manual'] = [
    'Generate the image that matches the description "{2}" and the given <image_input> {1}.',
    '<image_input> is the {1}. Draw an image that matches the description "{2}".',
    'Based on the <image_input> as {1}, generate an image with the description "{2}".',
    'Description: {2}. Instruction: Generate an image that matches the description and the input {1}.',
    'I would like you to build image of {2} and that matches this {1}',
    'Build an image of: {2}. It should match <image_input> {1}',
    'Context: <image_input> is {1} of the target image with the description {2}.\nTask: Generate the target image.',
]

PROMPT_DICT['controllable_image_gen']['gpt3'] = [
    "Produce an image in accordance with the provided {1} and description \"{2}\".",
    "Given <image_input> {1}, illustrate an image that aligns with the description \"{2}\".",
    "Create an image that resonates with the {1} and described by \"{2}\".",
    "Draw an image that is congruent with {1} and corresponds to \"{2}\".",
    "The {1} is provided as <image_input>. Compose an image in line with description \"{2}\".",
    "With <image_input> {1}, create an image consistent with the description \"{2}\".",
    "From {1}, generate an image that mirrors the description \"{2}\".",
    "Description provided: {2}. Instruction: Conjure an image aligning with the <image_input> {1}.",
    "Your task is to create an image of {2} that aligns with the <image_input> {1}.",
    "As per {1} <image_input>, craft an image that matches the description \"{2}\".",
    "Formulate an image that parallels {1} <image_input> and adheres to the description \"{2}\".",
    "Create an image that coincides with <image_input> {1} and adheres to \"{2}\".",
    "Based on <image_input> {1}, formulate an image reflecting the description \"{2}\".",
    # "Draw an image that is in tune with {1} and mirrors the description \"{2}\".",
    # "Given {1}, fashion an image that is congruous with the description \"{2}\".",
    # "Description provided: {2}. Task: Create an image that corresponds to the {1}.",
    # "Your assignment is to craft an image of {2} that is in sync with this {1}.",
    # "Craft an image: {2}. It should resonate with this {1}.",
    # "Given {1}, conceive an image that aligns with the narrative \"{2}\".",
    # "Formulate an image inspired by {1}, reflecting the depiction \"{2}\".",

    # "Using <image_input> {1}, make an image of \"{2}\".",
    # "Design an image of \"{2}\" from {1}.",
    # "Craft \"{2}\"'s image using {1}.",
    # "{1} to \"{2}\" image conversion.",
    # "Shape an image of \"{2}\" with {1}.",
    # "From {1}, illustrate \"{2}\".",
    # "Render \"{2}\"'s image from {1}.",
    # "{1} is provided. Depict \"{2}\".",
    # "Given {1}, manifest \"{2}\" in an image.",
    # "Create \"{2}\"'s image from {1}.",
    # "Illustrate \"{2}\" using {1}.",
    # "Depict \"{2}\" using {1} for reference.",
    # "Use <image_input> {1} to design \"{2}\"'s image.",
    # "Develop \"{2}\"'s image using {1}.",
    # "Construct \"{2}\"'s image from {1}.",
    # "Create \"{2}\"'s image from {1}.",
    # "{1} to \"{2}\" translation in image.",
    # "From <image_input> {1}, draw an image of \"{2}\".",
    # "{1}-based image for \"{2}\".",
    # "Use {1} to create an image of \"{2}\".",
]

PROMPT_DICT['controllable_image_gen_comb']['manual'] = [
    # There are few `#1` in original data
    'Generate the image that matches the description "{2}" from the combination of visual conditions {1} and {COND2}.',
    'Based on the {1} and {COND2}, draw an image that matches the description "{2}".',
    'From the {1} and {COND2}, Generate an image with the description "{2}".',
]

PROMPT_DICT['controllable_image_gen_reverse']['manual'] = [
    'Generate the visual condition {1} from the given image that matches the description " {2} ".',
    'Draw the {1} that matches the description " {2} " and the given image.',
    'Generate the {1} with the description " {2} " from the image.',
]

PROMPT_DICT['framenet']['manual'] = [
    'This is an image of an indoor scene. What are the surface normals of this image?',
    'Generate an image depicting the surface normal at each pixel of the given image of an indoor scene.', 
    'Given this image of an indoor scene, generate an image representing the orientation of the surface at each pixel.',
]

VIMA_ACTION_DESC = """\
as a sequence of sentences with the format: \
"step: start-action from ( a1 b1 r1 ) and end-action to ( a2 b2 r2 )"\
"""
PROMPT_DICT["vima"]["original"] = [
    f"Forecast the next actions {VIMA_ACTION_DESC} based on the agent's initial observation",
    f"Anticipate next moves {VIMA_ACTION_DESC} based on the agent's initial observation",
    f"Estimate subsequent actions {VIMA_ACTION_DESC} using the agent's initial observation",
    f"Compute next actions {VIMA_ACTION_DESC} based on the agent's initial observation",
    f"Propose following moves {VIMA_ACTION_DESC} based on the agent's initial observation",
    f"Predict forthcoming actions {VIMA_ACTION_DESC} based on the agent's initial observation",
    f"Project following actions {VIMA_ACTION_DESC} based on the agent's initial observation",
    f"Speculate forthcoming actions {VIMA_ACTION_DESC} based on the agent's initial observation",
    f"Determine next moves {VIMA_ACTION_DESC} based on the agent's initial observation",
    f"Surmise subsequent actions {VIMA_ACTION_DESC} based on the agent's initial observation",
    f"Approximate following actions {VIMA_ACTION_DESC} based on the agent's initial observation",
    f"Estimate next moves {VIMA_ACTION_DESC} based on the agent's initial observation",
    f"Guess next actions {VIMA_ACTION_DESC} based on the agent's initial observation",
    f"Formulate next actions {VIMA_ACTION_DESC} based on the agent's initial observation",
    "What is the action plan? Consider the observations", "Predict the actions given images",
]

PROMPT_DICT["vima_task_only"]["original"] = [
    "Predict a sequence of actions to achieve the goal",
    "Predict a sequence of actions to complete the goal",
    "Predict a sequence of actions to achieve the prompt",
    "Predict a sequence of actions to complete the prompt",
    f"Forecast the next actions to achieve the goal",
    f"Anticipate next sequence of moves to complete the goal",
    f"Propose following moves to achieve the goal",
    f"Predict forthcoming actions to finish the goal",
    f"Foresee subsequent actions to achieve the goal",
    f"Project following sequence actions to achieve the prompt",
    f"Assess subsequent moves", f"Expect next actions",
    f"Speculate forthcoming actions", f"Determine next moves",
    f"Estimate next moves", f"Guess next actions", f"Evaluate following moves",
    f"Formulate next actions",
]

PROMPT_DICT["vima_bboxes"]["original"] = [
    "bounding boxes of objects",
    "the bounding boxes' upper left and lower right corners of objects",
    "the y1, x1, y2, x2 coordinates of objects",
    "the y1, x1, y2, x2 regions of objects",
]

POINTNAV_TARGET_DESC = """\
{#OBS}with the discrete action space "{#action_space}"\
"""
POINTNAV_OBS_DESC = """\
based on agent's histories with the format \
"observation ( x z y ) action"\
"""
POINTNAV_DESC = f"{POINTNAV_TARGET_DESC} {POINTNAV_OBS_DESC}"
NEXT_ACTION_PROMPT = [
    f"Foresee the next action",
    f"Anticipate the following action",
    f"Compute the subsequent action",
    f"Estimate the next action",
    f"Propose the next move",
    f"Predict the following action",
    f"Calculate the next action",
    f"Forecast the next action",
    f"Conjecture the subsequent action",
    f"Project the next move",
    f"Estimate the forthcoming action",
    f"Assess the next action",
    f"Surmise the next action",
    f"Calculate the following move",
    f"Predict the subsequent action",
    f"Determine the next action",
    f"Approximate the next move",
    f"Guess the following action",
    f"Formulate the next action",
]
PROMPT_DICT["next_action_prediction"]["manual"] = NEXT_ACTION_PROMPT


PROMPT_DICT["habitat_pointnav"]["original"] = [
    _ + f" {POINTNAV_DESC}" for _ in NEXT_ACTION_PROMPT
]

PROMPT_DICT["habitat_pointnav_qa"]["original"] = [
    'Is the prediction "{}" correct?',
    'Should the agent take "{}" action?',
    'The agent is taking "{}". Do you recommend?',
    'Agent is going to "{}". Will this action lead to reach the goal?',
    'Judge if the action is "{}".',
]

PROMPT_DICT["embodied_goal_prediction"]["manual"] = [
    "Task: {} Given the start state as image, generate the end state image.",
    'Given the initial observation and prompt "{}", predict the goal image.',
    "Task: {} Predict the observation of the end state based on the initial observation.",
    "The robot is tasked '{}' at the current state. Predict the goal state.",
    "What does the goal image look like given the current robot RGB observation and prompt '{}'?",
    "Imagine what the final image state is after the robot complete the task '{}'",
    "Imagine you are the robot in this scene. Your task is: {} Generate the goal image after the task completed.",
    'What is the final frame after completing the task: "{}"',
    # image edit prompt
    "Build a new image based on the input image and these instructions: {}",
    'Apply the instruction "{}" on the given image.',
    "Generate a picture where '{}'",
    "Edit: {} Apply the edit to the image.",
]


PROMPT_DICT["manipulation_action_prediction"]["manual"] = [
    "".join([b, " ", a.lower() if lower else a, ".\n" if new_line else "."]) for b, lower, new_line in [
        ("Action space: {ACTION}.\nInstruction: {PROMPT}.\nTask:", False, True),
        ('The robot with the action space {ACTION} is doing the task "{PROMPT}".', False, False),
        ('With the continuous action space {ACTION}, the robot is doing the manipulation task "{PROMPT}."', False, False),
        ('The manipulator is tasked for "{PROMPT}". Given the action space {ACTION},', True, False),
        ('Imagine you are the robot in the scene with the action space {ACTION}. '
         'Based on the prompt "{PROMPT}" and current observation,', True, False),
    ]
    for a in NEXT_ACTION_PROMPT
]

PROMPT_DICT["embodied_history_prompt"]["manual"] = [
    "History: {H}", "Context: {H}",
]

PROMPT_DICT["embodied_task_caption"]["manual"] = [
    "Given the demonstration, what is the task?",
    "I want to write a prompt for this task, can you give me a short one?",
    "Provide me a short prompt for this robot demonstration",
    "What language instruction can be conditioned on this video?",
    "What is the robot doing?",
    "Decompose the task into languages.",
    "Think about a short task description based on the video.",
]

PROMPT_DICT['audio_generation']['original'] = [
    'Synthesize the sound based on the description "{}"',
]

PROMPT_DICT['audio_generation']['manual'] = [
    '{}',
    'Generate the sound based on the description: {}',
    'What might \"{}\" sound like?',
    'What is the sound of {}',
    'Create the noise of: {}',
    "Event: {} Generate a noise this event could plausibly make."
]

PROMPT_DICT['audio_generation']['gpt3'] = [
    "Generate the sound in accordance with the description '{}'.",
    "Create audio as described in '{}'.",
    "Develop a sound that fits the description '{}'.",
    "Produce the sound that matches the description '{}'.",
    "Formulate sound based on the given description '{}'.",
    "Construct a sound representation according to '{}'.",
    "Design a sound as characterized by '{}'.",
    "Generate audio corresponding to the details in '{}'.",
    "Create a sound consistent with the description '{}'.",
    "Develop audio according to the provided description '{}'.",
    "Fabricate a sound as per the given details '{}'.",
    "Create sound that fits the guidelines '{}'.",
    "Formulate a sound following the instructions '{}'.",
    "Craft audio as specified in '{}'.",
    "Devise a sound based on the information '{}'.",
    "Engineer a sound output matching '{}'.",
    "Produce audio that aligns with '{}'.",
    "Render a sound that is consistent with '{}'.",
    "Model a sound that corresponds with the description '{}'.",
    "Generate a sound that adheres to the description '{}'.",
]

PROMPT_DICT['audio_generation_no_text_no_history']['manual'] = [
    'Synthesize random sound.', 'Develop audio.',
    "Create sound.", 'Generate random audio',
    "Provide me some random audio.",
]

PROMPT_DICT['audio_generation_no_text_has_history']['manual'] = [
    "Create sound based on what you heard.",
    "Generate audio based on the audio history.",
    "Produce sound corresponding to the context.",
    "Continue to generate the audio.", "Continue to generate the sound.",
    "Continue to synthesize the audio.", "Continue to synthesize the sound.",
    "Construct an audio continuation based on the given samples.",
    "Mimic the sounds you've just been exposed to.",
    "Extend the audio narrative using the given audios.",
    "Listen to the preceding audio and build upon it.",
    "Fabricate an audio sequence in tune with the prior soundscapes.",
    "Keeping the previous tones in mind, what comes next in this audio tale?",
    "How would the next chapter of this sound story go?",
    "Based on the auditory history, weave a new sonic tapestry.",
    "Following the previous auditory journey, create its successor.",
]


PROMPT_DICT['audio_text_to_speech_generation']['original'] = [
    'Read "{}"',
]


PROMPT_DICT['audio_text_to_speech_generation']['manual'] = [
    'Read texts: "{}".',
    'Speak: {}.',
    'Please generate sounds that read the script "{}".',
    "Produce a spoken version of the text: {}.",
    "Generate spoken words from the text '{}'.",
    "Craft an auditory reading of '{}' for me.",
    "Could you please vocalize the text: '{}'?",
    "Convert '{}' into speech.",
    'Turn this written text into spoken words: "{}".',
    "Orally present the text '{}' for me.",
    "Articulate the following text as speech: {}.",
    'Speak out the text "{}" for me.',
    "Mimic human speech with the text '{}'.",
    "Let's hear how '{}' would sound when read aloud.",
]

PROMPT_DICT['audio_speech_to_text_generation']['original'] = [
    'Transcribe the audio to text.',
]

PROMPT_DICT['audio_speech_to_text_generation']['manual'] = [
    'Turn the spoken words into text.',
    'Please generate text that transcribes the audio.',
    'Could you transform the spoken language into text?',
    'Could you provide a written version of the audio content?',
    'Convert the audio into text.',
    'Transcribe the spoken language for me.',
    'Can you put into text what the speaker exactly said in the audio?',
]

# detailed description
PROMPT_DICT['music_caption']['original'] = [
    "Describe the music."
]

PROMPT_DICT['music_caption']['manual'] = [
    "What kind of narrative does this piece of music suggest to you?",
    "How would you describe the tone and feeling evoked by this sound?",
    "From an auditory perspective, what elements stand out in this music piece?",
    "Analyze the music in this audio.",
    "What sonic landscapes does this music evoke in your mind?",
    "Discuss the music in detail.",
    "How's the music? Long answer:",
    "Can you break down the components of this music and suggest possible media scenarios where it might be used?",
    "In your own words, could you detail the musical components and atmosphere this song creates?",
]


PROMPT_DICT['music_generation']['original'] = [
    "Generate the music based on the description '{}'",
]

PROMPT_DICT['music_generation']['manual'] = [
    "Create an auditory representation that matches the description ' {} '",
    'Craft a musical piece inspired by the details provided in " {} "',
    "Based on the given description '{}', produce a corresponding piece of music.",
    "Can you realize this musical description '{}' in the form of an audio track?",
    "Translate the described musical elements in '{}' into an actual piece of music.",
    "Turn the textual information in '{}' into a musical composition.",
    "Could you create an audio expression that mirrors the details in '{}'",
    "Convert the written narrative '{}' into a corresponding musical piece.",
    "Use the description '{}' as the blueprint for a new piece of music.",
    'Compose an audio track that embodies the spirit and details of "{}"',
]


PROMPT_DICT['audio_caption']['original'] = [
    'Give a short description of this audio.',
    'Caption the audio with a short description.',
    'Write a short description of this audio.',
    'Describe the audio in short.',
    'Write a short caption describing this noise.',
    'In brief, what is this sound?'
]


PROMPT_DICT['Audio_Tagging_Scene']['manual'] = [
    'Only identify the scene present in the audio.',
    'What scene does this audio depict? Please provide me the name of the scene only.',
    'This audio depicts a scene of a',
    'This sound is set in a',
    'The scene of this audio is a',
    'What kind of place is this? Short answer:',
    'Please give me a short answer of what scene is in the audio?',
    'This sound shows the scene of a',
]

PROMPT_DICT['Audio_Tagging_Scene']['gpt3'] = [
    'Listen and just point out the scenario represented in the audio.',
    'What situation is portrayed in this audio? Please only give the name of the scenario.',
    'The audio reflects a situation of a',
    'The setting of this audio is a',
    'The situation illustrated in this audio is a',
    'What sort of location is this? Brief answer:',
    'Could you provide a concise description of the scene depicted in the audio?',
    'Solely describe the tableau in the audio.',
    'What tableau is depicted in this sound? Only name the scene, please.',
    'The audio portrays the tableau of a',
    'The tableau of this audio is set in a',
    'This audio illustrates a tableau of a',
    'Kindly give a short answer describing the scene in the audio.',
    'Identify solely the setting present in what you are listening.',
    'What scenario is this audio illustrating? Only provide the scene\'s name.',
    'The audio represents a scenario of a',
    'What you listened situates in a',
    'The setting depicted in this audio is a',
    'Can you provide a brief description of the audio\'s scene?',
    'This sound portrays the setting of a',
    'Focus only on identifying the scene in the audio.',
    'What does this audio illustrate? Please provide only the scene\'s name.',
    'This audio showcases a setting of a',
    'What you are hearing is established in a',
    'This audio projects a scene of a',
    'What you hear presents the setting of a',
    'Just pinpoint the scene captured in what you listened.',
]

PROMPT_DICT['audio_tagging']['original'] = [
    'Identify the main sounds present in the given audio clip with a few words.',
    'What is this in the audio clip? Please provide a short answer.',
    'What is this in the audio? Short answer:',
    'Please provide a short answer. What does audio contain?',
]

PROMPT_DICT['audio_tagging']['gpt3'] = [
    'Analyze the given audio clip and identify its primary audible elements with a few words.',
    'Determine the primary sounds featured in the provided sound in a short answer.',
    'Discern the major sounds captured in this audio sample in a few words.',
    'Recognize the key audios you hear in this sound clip. Please provide a short answer.',

    "Briefly describe the principal sounds in the audio clip.",
    "What can be heard in this audio clip? Give a brief response.",
    "Which are the main sounds heard in the audio? Respond in a few words.",
    "In a few words, specify the sounds in the given audio clip.",
    "Quickly identify the primary noises in the audio.",
    "What sounds can be identified in this audio clip? Short answer, please.",
    "Could you briefly describe what's audible in the audio clip?",
    "Short response: What's the sound content of the audio?",
    "What are the primary audible elements in the clip? Please keep the response short.",
    "Identify the predominant sounds in the audio. Short response needed.",
    "Briefly outline what you hear in the audio clip.",
    "Describe the main auditory elements in the clip in a few words.",
    "List the key sounds you can hear in the audio clip.",
    "In a few words, what are the primary noises in the audio?",
    "What's the main sound in the clip? Briefly explain.",
    "Please shortly define the sounds in the given audio clip.",
    "What sounds stand out in this audio? Please answer briefly.",
    "Please give a succinct description of the sounds in the audio.",
    "Short answer: What noises are prominent in the audio clip?",
    "Briefly detail the main sounds found in the audio.",
]

PROMPT_DICT['visual_sound_localization']['original'] = [
    "Identify the region of the sound sources in the given image, return a sequence of bounding boxes"
    " that locate each sound source.",
    "Find the bounding box's upper left and lower right corners of the sound sources in the given image",
    "Locate the bounding boxes of the sound sources in the given image",
    "What in this image is likely producing these sounds? Identify them by producing bounding boxes for each source.",
    "Find the y1, x1, y2, x2 coordinates of the sound sources in the given image",
    "Identify the locations of the sound sources in the given image, indicated by the coordinates of the top-left and bottom-right corners (y1, x1, y2, x2).",
]

PROMPT_DICT['video_captioning']['original'] = [
    'What does the video describe ?',
    'Caption the video with a short description.',
    'Write a short description of this video.',
    'Describe the video with a short description.',
    'Write a brief caption describing the video.'
]

PROMPT_DICT['video_captioning']['gpt3'] = [
    "What is the video portraying in a short description? ",
    "Can you tell what the video is depicting in short?",
    "Can you briefly explain what is the video illustrating ?",

    'Give a brief summary of the video.',
    'In a short statement, caption the video.',
    'What is the video showing? Please describe in short.',
    'Describe the content of the video briefly.',
    'Please summarize the video in a few words.',
    'What is the main theme of the video? Please write briefly.',
    'What\'s the video about? Give a concise description.',
    'Briefly, what does this video present?',
    'In brief terms, explain the content of the video.',
    'Give a succinct summary of what the video shows.',
    'In a few words, describe the video.',
    'Can you briefly caption this video?',
    'Please give a quick summary of the video.',
    'What\'s the key takeaway from the video? Briefly describe.',
    'Briefly, what is this video all about?',
    'What is the essence of the video? Describe in a sentence.',
    'Write a short caption that summarizes the video.',
    'Please provide a brief caption for the video.',
    'In short, what does this video depict?',
    'How would you briefly summarize the video?',
    'What\'s the brief overview of the video?',
    'Provide a concise narrative of the video.',
    'Can you encapsulate the video\'s content in a few words?',
    'Briefly capture the essence of the video.',
    'Give a short explanation of what the video is about.',
    'What\'s the main point of the video? Provide a brief description.',
    'In a nutshell, what is the video presenting?',
    'What\'s the short summary of the video?',
    'Write a brief description encapsulating the video.',
]

PROMPT_DICT['video_action_tagging_ssv2']['original'] = [
    'What fine-grained human hand gestures is being performed in this video? Please provide a short answer.',
    'What fine-grained human hand gestures is going on in the video? Short answer:',
    'Please provide a short answer. What fine-grained human hand gestures is going on in the video?',
]

PROMPT_DICT['video_tagging']['original'] = [
    'What action is being performed in this video? Please provide a short answer.',
    'What activity is going on in the video? Short answer:',
    'Please provide a short answer. What activity is going on in the video?',
]

PROMPT_DICT['video_tagging']['manual'] = [
    'What is the salient activity of the video? Please provide a short answer.',
    'What are they doing in the video with short answer?',
    'What are they doing in this video? Short answer: ',
    'What action does this video depict? Please provide a short answer.',
    'Please provide a short answer. What action takes place in this video?',
    'What is the most prominent action in this video?',
    'Name the most prominent activity in this video with a short answer.',
]

PROMPT_DICT['video_tagging']['gpt3'] = [
    "What is happening in this video clip? Short answer:",
    "Can you describe the activity occurring in this video with a few words?",
    "What activity is shown in this video in short answer?",
    "What is the action demonstrated in this video? Please provide a short answer.",
    "What event is being depicted in this video? Please provide a short answer.",
    "Can you tell what's going on in this video with a few words?",
    "What kind of action does this video portray in short answer?",
    "What is the action happening in this video clip? Please provide a short answer.",
    "What is the specific action depicted in this video with a few words?",
    "What sort of action is the video showing? Short answer:",
    'Could you briefly tell me what\'s happening in this video?',
    'Give a short response: What\'s the main action in the video?',
    'What\'s the main event in the video? Respond briefly, please.',
    'What\'s the central activity in the footage? Please keep the answer short.',
    'Please succinctly describe the action in the video.',
    'In a few words, what is happening in this video?',
    'What is the video showing? Provide a short explanation.',
    'Please tell me in brief: What event is captured in this video?',
    'What\'s the significant action in the footage?',
    'Provide a brief answer about the main event in the video.',
    'What is being carried out in this video? Briefly describe.',
    'In few words, what\'s the primary event in the video?',
    'What does the footage principally depict? A short answer, please.',
    'Could you summarise the main action in the video?',
    'In brief, what\'s the principal event unfolding in this video?',
    'What are the subjects in the video doing? Give a short answer.',
    'What\'s the main occurrence in the video? Please respond briefly.',
    'Please give a succinct account of the video\'s main activity.',
    'Identify the central action in this video in a few words.',
    'What\'s the crucial event in the footage? Respond concisely, please.',
    'Describe the key activity in this video briefly.',
    'In a nutshell, what\'s unfolding in the video?',
    'Please briefly describe the main occurrence in the video.',
    'What\'s the significant event in the video? Provide a succinct response.',
    'Describe in few words, the main action captured in this video.',
    'What action is the video mainly highlighting? Please keep your answer short.',
    'Can you briefly explain the primary activity in this video?',
    'What\'s the primary focus of the video in terms of activity? Respond shortly, please.',
    'In a concise answer, what\'s the video principally showing?',
    'What is the subject doing in the video? Provide a brief response.',
]

PROMPT_DICT['video_qa_short_prompt']["manual"] = [
    "{} A short answer to the question is",
    "{} Short answer:",
    "Please provide a short answer. {}",
    "Given the video, answer the following question with no more than three words. {}",
    "{} Based on the video, respond to this question with a short answer:",
    "Use the provided video to answer the question: {} Provide your answer as short as possible:",
    'The question "{}" can be answered using the video. A short answer is',
    '{} Answer the question as briefly as possible.',
]

PROMPT_DICT['audio_qa_short_prompt']["manual"] = [
    "{} A short answer to the question is",
    "{} Short answer:",
    "Please provide a short answer. {}",
    "Given the audio, answer the following question with no more than three words. {}",
    "{} Based on the audio, respond to this question with a short answer:",
    "Use the provided audio to answer the question: {} Provide your answer as short as possible:",
    'The question "{}" can be answered using the audio. A short answer is',
    '{} Answer the question as briefly as possible.',
]

PROMPT_DICT['audio_visual_qa_short_prompt']["manual"] = [
    "{} A short answer to the question is",
    "{} Short answer:",
    "Please provide a short answer. {}",
    "Answer the following question with no more than three words. {}",
    "{} Respond to this question with a short answer:",
    '{} Answer the question as briefly as possible.',
]


PROMPT_DICT['video_anticipation']['original'] = [
    'What action is most likely to happen next in this video? A short answer to the question is',
    'What action is most likely to happen next in this video? Short answer:',
    'Please provide a short answer. What action is most likely to happen next in this video?',
    'What action is most likely to happen next in this video? Respond to this question with a short answer:',
    'What action is most likely to happen next in this video? Answer the question as briefly as possible.'
]

PROMPT_DICT['video_anticipation']['gpt3'] = [
    "What do you think will be the next action in this video?  A short answer to the question is",
    "What is anticipated to happen next in this video? Short answer:",
    "Can you foresee the next action in this video with a few words?",
    "What's the most probable next move in this video with a few words?",
    "What event is expected to occur next in this video? Short answer:",
    "Can you predict the next event in this video with a short answer?",
    "What's the expected next scene in this video with a few words?",
    "What do you presume will happen next in this video? Respond to this question with a short answer:",
    "What is the forthcoming action in this video likely to be in short answer?",
    'Predict the next action in the video briefly.',
    'What\'s the most probable next action in this video? Short answer, please.',
    'Please give a concise prediction of the next action in this video.',
    'What do you think will happen next in this video? Please answer shortly.',
    'In a few words, what is likely to happen next in the video?',
    'What action might unfold next in the video? Briefly state.',
    'Give a short prediction of the next action in this video.',
    'Please provide a brief response. What\'s the likely next event in the video?',
    'In a nutshell, what\'s the next likely action in this video?',
    'What is the video likely to show next? Please provide a short answer.',
    'What comes next in the video? Respond briefly.',
    'Give a brief prediction about the next scene in the video.',
    'Can you briefly describe the possible next action in the video?',
    'What\'s likely to occur next in the video? Short response, please.',
    'Briefly predict the next happening in the video.',
    'What\'s the anticipated next action in the video? Please answer briefly.',
    'Please answer in short. What is the next likely event in this video?',
    'What do you anticipate to be the next action in the video? Briefly explain.',
    'Briefly predict what could happen next in the video.',
    'What\'s the expected next occurrence in the video? Give a short answer.',
]

PROMPT_DICT['video_action_localization']['original'] = [
    'Given the temporal context from the video, detect all of the humans performacing actions in the image.'
]

PROMPT_DICT['video_action_localization']['manual'] = [
    'Locate all humans performing actions in the image considering the video context.',
    'Find regions of all of the humans performing actions in the image considering the temporal context provided by the video.',
    'There are one or more humans performing actions in this image. Considering the temporal context from the video, find the regions corresponding to each of them.',
]

PROMPT_DICT['video_tracking']['original'] = [
    'Anticipate the object\'s next location from all previous images and the location of the object in those frames: {}.',
    'Given all previous location of the object: {}, estimate the object\'s next location.',
]

PROMPT_DICT['video_tracking']['gpt3'] = [
    'Project the next possible location of the object, based on its positions in all the previous images: {}.',
    'Predict where the object will be located next, taking into account all previous images and its location within them: {}.',
    'Estimate the object\'s future location using all past images and the object\'s position within these frames: {}.',
    'Use all previous images and the object\'s position in those frames to infer its next location: {}.',
    'Expect the object\'s next location by taking into account all prior images and the position of the object in those frames: {}.',
    'Determine the object\'s next location based on its placement in all preceding images: {}.',

    'Predict the object\'s next position based on its previous locations in the frames: {}.',
    'Based on the object\'s previous locations: {}, foresee its next position.',
    'Please project the object\'s next location using its past locations in these frames: {}.',
    'Utilizing the object\'s past locations: {}, can you guess its upcoming location?',
    'From the previous images and object locations: {}, extrapolate the object\'s next location.',
    'Given the object\'s past locations in these frames: {}, predict its next spot.',
    'Project the next location of the object based on its previous locations: {}.',
    'By observing the object\'s past locations: {}, can you deduce its next location?',
    'Based on the object\'s prior positions: {}, estimate its future location.',
    'Foretell the object\'s next location using its historical positions from these frames: {}.',
    'With the object\'s previous locations: {} as reference, anticipate its next placement.',
    'Can you project the next spot of the object given its prior positions: {}?',
    'Considering the object\'s previous positions in these frames: {}, predict its subsequent location.',
    'Speculate the next position of the object, given its past positions: {}.',
    'From its past locations: {}, can you predict where the object might be next?',
    'Given the object\'s trajectory in these frames: {}, foresee its next location.',
    'Guess the next location of the object from its prior locations: {}.',
    'From the previous object locations: {}, infer the next probable location.',
    'Based on its past positions: {}, can you anticipate the object\'s next location?',
    'Where do you foresee the object next, given its previous positions: {}?'
]

PROMPT_DICT['video_prediction']['original'] = [
    'Generate the " {1} " video frame from previous video frames and the locations of the object: " {2} ".',
]

PROMPT_DICT['video_prediction']['gpt3'] = [
    'From prior video frames and the object\'s locations: " {2} ", create the " {1} " video frame.',
    'Use previous video frames and the object\'s locations: " {2} " to produce the " {1} " video frame.',
    'By leveraging previous video frames and the object\'s locations: " {2} ", synthesize the " {1} " video frame.',
    'With the use of prior video frames and the object\'s locations: " {2} ", create the " {1} " video frame.',
    'Construct the " {1} " video frame drawing upon previous video frames and the object\'s locations: " {2} ".',
    'Develop the " {1} " video frame by using previous video frames and the object\'s locations: " {2} ".',
    'Utilize prior video frames and the locations of the object: " {2} " to generate the " {1} " video frame.',
]

PROMPT_DICT['video_multipoint_tracking']['original'] = [
    'Predict the "y x" location of the visible points in the image from the locations of them in previous frames',
]

PROMPT_DICT['video_multipoint_tracking']['gpt3'] = [
    'Anticipate the "y x" positioning of the visible points in the image based on their locations from previous frames',
    'Forecast the "y x" coordinates of the visible points in the image using the locations from the past frames as a reference',
    'Estimate the "y x" location of the visible points in the image, utilizing their previous frames\' positions',
    'Use the locations of the points in the previous frames to predict their "y x" coordinates in the image if they are visible',
    'Determine the "y x" location of the visible points in the image by referencing their positions in past frames',
    'Using the points\' locations from prior frames, foresee their "y x" position in the image if they are visible',
    'Infer the "y x" coordinates of the visible points in the image from their past frame locations',
    'Predict the "y x" positioning of the visible points in the image using their past frame locations as guidance',
]

PROMPT_DICT['video_multipoint_tracking_special_token']['original'] = [
    'Predict the "y x" location of the visible points in <image_input> from the locations of them in previous frames',
]

PROMPT_DICT['video_multipoint_tracking_special_token']['gpt3'] = [
    'Anticipate the "y x" positioning of the visible points in <image_input> based on their locations from previous frames',
    'Forecast the "y x" coordinates of the visible points in <image_input> using the locations from the past frames as a reference',
    'Estimate the "y x" location of the visible points in <image_input>, utilizing their previous frames\' positions',
    'Use the locations of the points in the previous frames to predict their "y x" coordinates in <image_input> if they are visible',
    'Determine the "y x" location of the visible points in <image_input> by referencing their positions in past frames',
    'Using the points\' locations from prior frames, foresee their "y x" position in <image_input> if they are visible',
    'Infer the "y x" coordinates of the visible points in <image_input> from their past frame locations',
    'Predict the "y x" positioning of the visible points in <image_input> using their past frame locations as guidance',
]

PROMPT_DICT['video_question_generation']['original'] = [
    'Generate a question about the video that can be answered with " {} ".',
    'Create a query concerning the video that " {} " could be a suitable answer for.',
    'Formulate a question about the video to which " {} " can serve as a response.',
    'Develop a question related to the video that " {} " would be the appropriate answer for.',
    'Conceive a question based on the video where " {} " fits as the answer.',
    'Design a question tied to the video that would be correctly answered by " {} ".',
    'Given the video, generate a question whose answer is: " {} ". Question:',
    'Based on images, provide a question with the answer: " {} ". Question:',
    'Given the video, create a question for which the answer is " {} ".',
    'From the few images provided, craft a question that leads to the reply: " {} ". Question:'
]

PROMPT_DICT['audio_question_generation']['original'] = [
    'Generate a question about the audio that can be answered with " {} ".',
    'Create a query concerning the audio that " {} " could be a suitable answer for.',
    'Formulate a question about the audio to which " {} " can serve as a response.',
    'Develop a question related to the audio that " {} " would be the appropriate answer for.',
    'Conceive a question based on the audio where " {} " fits as the answer.',
    'Design a question tied to the audio that would be correctly answered by " {} ".',
    'Given the audio, generate a question whose answer is: " {} ". Question:',
    'Based on audio, provide a question with the answer: " {} ". Question:',
    'Given the audio, create a question for which the answer is " {} ".',
    'From the audio provided, craft a question that leads to the reply: " {} ". Question:'
]

PROMPT_DICT['3d_object_detection']['original'] = [
    'Locate all objects in 3D format in the image.',
    'What is the projected 3D center, virtual depth, log-normalized box size and rotation of all objects in the image.',
    'Detect all objects in 3D using projected 3D center, virtual depth, log-normalized box size and rotation in the image.',
    'Locate all objects in 3D using projected 3D center, virtual depth, log-normalized box size and rotation in the image.',
]

PROMPT_DICT['optical_flow']['original'] = [
    'Describe the flow of objects from the input image to the reference image.',
]

PROMPT_DICT['optical_flow']['manual'] = [
    'Optical flow',
    'Please provide the optical flow of given images.',
    'Illustrate the object movement.',
    'Visualize the optical flow.',
    'Render optical flow.',
    'Draw optical flow.',
    'What does the optical flow of these images look like?',
    'Generate an image illustrating the optical flow from the input image to the reference image.',
    'Construct an image that depicts the transition of objects from the input to the reference image.',
    'Visualize the sequence of object displacement from the input to the reference image in a single frame.',
    'Can you create an image that encapsulates the flow of objects from the initial to the target image?',
    'Provide an image that encapsulates the shift in objects when transitioning from the input to the reference image.',
]


PROMPT_DICT['sci_caption_ocr']['manual'] = [
    'The image contains "{}".',
    'The optical character recognition of the figure is "{}".',
    'The OCR of this image is "{}".',
    'The text in this image is: {}.',
]

PROMPT_DICT['sci_caption_mention']['manual'] = [
    'In a scientific paper, the author mentions this figure by "{}".',
    'A scientific paper mentions "{}" for the image.',
    'A scholarly document contains the text: {}.',
    'The author said "{}" in a scientific paper.',
    'The author of the paper elaborates this figure by "{}".',
    "In a scientific paper, the author mentions <image_input> with:\n{}\n",
    "This figure comes from a scientific document. The document references the figure by {}"
]

PROMPT_DICT['sci_caption']['manual'] = [
    'Write a description of <image_input> that could server as the figure caption in the paper.',
    "Guess what the caption of this figure would be in the paper.",
    "What could the figure caption have been?",
    'What would the caption of this figure in the paper be?',
    'What is the caption for this image in the paper?',
    "Construct a caption that could have been used for this figure in the paper.",
    "Based on this information, what could the figure's caption have been?"
]

PROMPT_DICT['image_edit']['manual'] = [
    "Build a new image based on the input image and these instructions: {}",
    'Apply the instruction "{}" on the given image.',
    "Generate a picture where {}",
    "Edit: {}. Apply the edit to the image.",
    "What would it look like if the image was modified by \"{}\"?",
    'Transform the image with following this instruction: {}',
    "Alter the image using the subsequent directive: {}",
    "Apply the ensuing instruction to modify the image: {}",
    "Adjust the image guided by the following direction: {}",
    "With the next instruction, effect changes on the image: {}",
    "Reform the image based on the following guideline: {}",
]

PROMPT_DICT['image_edit']['gpt3'] = [
    "Create a new image by following these instructions: {}",
    "Modify the provided image with the guidance of this instruction: {}",
    "Fabricate an image with the condition: {}",
    "Edit to include: {}. Apply this change to the image.",
    "How would the image appear if the modification of \"{}\" was applied?",
    "Revamp the image by adhering to this instruction: {}",
    "Change the image according to the ensuing rule: {}",
    "By using the following instruction, alter the image: {}",
    "Tweak the image by following this particular directive: {}",
    "Under the guidance of the next instruction, modify the image: {}",
    "Amend the image by using the following principle: {}",
    "Morph the image by implementing the following rule: {}",
    "By applying this instruction: {}, reshape the image.",
    "Render a new image from the existing one with these instructions: {}",
    "Enforce the following directive on the image to modify it: {}",
    "Following this direction: {}, customize the image.",
    "Manipulate the image using this advice: {}",
    "Remold the image in accordance with the following guide: {}",
    "With the subsequent command, reshape the image: {}",
    "Reconfigure the image according to the following blueprint: {}",

    "With instructions {}: create a new image.",
    "Alter the image: {}.",
    "Render an image: {}.",
    "Apply {}: edit the image.",
    "Image after modification: {}?",
    "Revamp image using: {}.",
    "Change image with: {}.",
    "Alter image via: {}.",
    "Tweak image with: {}.",
    "Modify image under: {}.",
    "Amend image with: {}.",
    "Morph image using: {}.",
    "Reshape image applying: {}.",
    "Render new image from: {}.",
    "Enforce on image: {}.",
    "Customize image under: {}.",
    "Manipulate image via: {}.",
    "Remold image with: {}.",
    "Reshape image with: {}.",
    "Reconfigure image via: {}.",
]


PROMPT_DICT['multi_turn_image_edit']['manual'] = [
    'You are doing an image editing task where you apply a sequence of user-supplied image editing'
    ' instructions. Instructions should be applied in order. Your instructions are {}',
    'Please modify this picture following these instructions: {}',
    "Help me change this picture. Apply these edits: {}, in order, and generate the resulting image.",
    "Here are some image editing instructions: {}, apply them to the image."
]


PROMPT_DICT['multi_turn_image_edit']['gpt3'] = [
    "You've been tasked with an image editing assignment where you'll be implementing a set of user-given image editing instructions. The instructions, which are {}, should be executed sequentially.",
    "You are to carry out an image editing task. Please adhere to the instructions provided in the sequence {}. Apply them in order.",
    "Perform an image editing task. Use these user-supplied instructions in the given order: {}.",
    "This is an image modification task. Please apply these directions {} sequentially.",
    "For this image editing task, execute these user-specified steps {} in their given order.",
    "Your task is image editing, and you need to implement these ordered steps: {}.",
    "Carry out an image editing task by following these instructions: {}. Remember, the order matters.",
    "Alter this image according to these guidelines: {}. Make sure to follow the sequence.",
    "Edit this image by using the given instructions: {}. Apply them one after the other.",
    "Your job is to edit an image using these steps: {}. The sequence of application matters.",
    "In this image editing task, you need to follow these user-provided directives {} in the correct order.",
    "This image editing task requires applying these user-specified instructions: {} sequentially.",
    "You're tasked with an image editing job where you apply these instructions: {}. The sequence should be followed.",
    "For this image editing assignment, you'll need to apply these instructions: {}, in the order given.",
    "You've got an image editing job. Apply these provided instructions {} sequentially.",
    "An image modification task awaits you. The instructions are {}. Apply them in the specified order.",
    "This image needs editing. Use the following instructions: {}, but remember to follow the order.",
    "We have an image that needs adjusting. Please follow these directions: {}. Order is key.",
    "This image requires some alterations. Implement these instructions: {} sequentially.",
    "Here's an image editing task for you. Please apply these steps: {} in the right sequence.",
    "Apply user-supplied image editing steps in order: {}.",
    "Please follow these image editing instructions in sequence: {}.",
    "Use these instructions to edit the image: {}.",
    "Task: Image edit using instructions: {}.",
    "Follow these steps in order to edit the image: {}.",
    "Sequentially apply these image editing instructions: {}.",
    "Edit image using these instructions in sequence: {}.",
    "Implement these editing steps on the image: {}.",
    "In order, apply these instructions to edit the image: {}.",
    "Apply these user-given image editing steps sequentially: {}.",
    "Sequentially follow these steps to edit image: {}.",
    "Edit this image with these instructions: {}.",
    "Image editing task instructions: {}.",
    "Use these steps in order for image editing: {}.",
    "Your image editing instructions: {}.",
    "Task: Edit image using instructions: {}.",
    "Modify image using these sequential steps: {}.",
    "Apply instructions in order to edit the image: {}.",
    "Implement these sequential instructions on the image: {}.",
    "Follow instructions in sequence to edit image: {}."
]

# DEPRECATED
PROMPT_DICT['Prompt_Pretrain_View_Denoising']['original'] = [
    "In a spherical coordinate system centered on the object, define the camera location using polar and azimuth angles theta and phi, and radius r. Add missing details to a novel view of the object in the reference picture by applying the relative camera transformation (theta sin(phi) cos(phi) r)",
    "In a spherically centered coordinate system around the object, identify the camera's position using the polar angle (theta) and azimuth angle (phi), as well as the distance (r). Complete the image, it shows a new viewpoint of the object in the reference image obtained by utilizing the corresponding relative camera transformation (theta sin(phi) cos(phi) r)",
    "In an object-centered spherical coordinate system, determine the camera position by utilizing the polar angle (theta), azimuth angle (phi), and radial distance (r). Complete a distinct view of the object in the reference picture by executing the relative camera transformation (theta sin(phi) cos(phi) r)",
    "Using a spherical coordinate system that centers on the object, specify the camera's location through the polar angle (theta), azimuth angle (phi), and the radius (r). The reference image shows an object, generate the complete image of the object from a distinct camera viewpoint through the relative camera transformation (theta sin(phi) cos(phi) r)",
    "In a spherical coordinate system with the object at the center, locate the camera by employing the polar angle theta, azimuth angle phi, and distance r. Fill in the gaps of the image, which shows the object in the reference image from a different camera viewpoint by applying the relative camera transformation (theta sin(phi) cos(phi) r)",
    "The camera viewpoint is defined by the polar angle theta, the azimuth angle phi and the radius r, with the change in viewpoint represented using their differences. Finish the incomplete image of the reference object which shows a novel view after applying the camera transformation (theta sin(phi) cos(phi) r)",
    "The camera perspective is determined by the polar angle theta, the azimuth angle phi, and the radius r, with the change in viewpoint expressed through their respective differences. Create a complete image of the reference object after implementing the camera transformation (theta sin(phi) cos(phi) r)",
    "The camera's viewpoint is characterized by the polar angle theta, azimuth angle phi, and radial distance r, while the shift in perspective is denoted by their differences. Finish an image of the reference object by applying the camera transformation (theta sin(phi) cos(phi) r)",
    "The camera position is established by the polar angle theta, the azimuth angle phi, and the radius r, and the alteration in viewpoint is indicated by the differences between these values. Complete a new image of the reference object following the implementation of the camera transformation (theta sin(phi) cos(phi) r)",
    "The camera's orientation is dictated by the polar angle theta, the azimuth angle phi, and the distance r, with the modification in perspective signified by their corresponding differences. Add the missing elements to the image which shows the object in the reference picture after executing the camera transformation (theta sin(phi) cos(phi) r)",
]

PROMPT_DICT['Prompt_Pretrain_View_Generation']['original'] = [
    "In a spherical coordinate system centered on the object, define the camera location using polar and azimuth angles theta and phi, and radius r. Add missing details to a novel view of the object in the reference picture by applying the relative camera transformation (theta sin(phi) cos(phi) r)",
    "In a spherically centered coordinate system around the object, identify the camera's position using the polar angle (theta) and azimuth angle (phi), as well as the distance (r). Complete the image, it shows a new viewpoint of the object in the reference image obtained by utilizing the corresponding relative camera transformation (theta sin(phi) cos(phi) r)",
    "In an object-centered spherical coordinate system, determine the camera position by utilizing the polar angle (theta), azimuth angle (phi), and radial distance (r). Complete a distinct view of the object in the reference picture by executing the relative camera transformation (theta sin(phi) cos(phi) r)",
    "Using a spherical coordinate system that centers on the object, specify the camera's location through the polar angle (theta), azimuth angle (phi), and the radius (r). The reference image shows an object, generate the complete image of the object from a distinct camera viewpoint through the relative camera transformation (theta sin(phi) cos(phi) r)",
    "In a spherical coordinate system with the object at the center, locate the camera by employing the polar angle theta, azimuth angle phi, and distance r. Fill in the gaps of the image, which shows the object in the reference image from a different camera viewpoint by applying the relative camera transformation (theta sin(phi) cos(phi) r)",
    "The camera viewpoint is defined by the polar angle theta, the azimuth angle phi and the radius r, with the change in viewpoint represented using their differences. Finish the incomplete image of the reference object which shows a novel view after applying the camera transformation (theta sin(phi) cos(phi) r)",
    "The camera perspective is determined by the polar angle theta, the azimuth angle phi, and the radius r, with the change in viewpoint expressed through their respective differences. Create a complete image of the reference object after implementing the camera transformation (theta sin(phi) cos(phi) r)",
    "The camera's viewpoint is characterized by the polar angle theta, azimuth angle phi, and radial distance r, while the shift in perspective is denoted by their differences. Finish an image of the reference object by applying the camera transformation (theta sin(phi) cos(phi) r)",
    "The camera position is established by the polar angle theta, the azimuth angle phi, and the radius r, and the alteration in viewpoint is indicated by the differences between these values. Complete a new image of the reference object following the implementation of the camera transformation (theta sin(phi) cos(phi) r)",
    "The camera's orientation is dictated by the polar angle theta, the azimuth angle phi, and the distance r, with the modification in perspective signified by their corresponding differences. Add the missing elements to the image which shows the object in the reference picture after executing the camera transformation (theta sin(phi) cos(phi) r)",
]

PROMPT_DICT['Prompt_Pretrain_View_Generation_Special_Token']['original'] = [
    "In a spherical coordinate system centered on the object, define the camera location using polar and azimuth angles theta and phi, and radius r. "
    "Add missing details to a novel view of the object in <image_input> by applying the relative camera transformation (theta sin(phi) cos(phi) r)",

    "In a spherically centered coordinate system around the object, identify the camera's position using the polar angle (theta) and azimuth angle (phi), "
    "as well as the distance (r). Complete the image, it shows a new viewpoint of the object in <image_input> obtained by utilizing the corresponding "
    "relative camera transformation (theta sin(phi) cos(phi) r)",

    "In an object-centered spherical coordinate system, determine the camera position by "
    "utilizing the polar angle (theta), azimuth angle (phi), and radial distance (r). "
    "Complete a distinct view of the object in <image_input> by executing the relative camera transformation (theta sin(phi) cos(phi) r)",

    "Using a spherical coordinate system that centers on the object, "
    "specify the camera's location through the polar angle (theta), azimuth angle (phi), and the radius (r). "
    "The <image_input> shows an object, generate the complete image of the object from a distinct camera viewpoint "
    "through the relative camera transformation (theta sin(phi) cos(phi) r)",

    "In a spherical coordinate system with the object at the center,"
    " locate the camera by employing the polar angle theta, azimuth angle phi, and distance r. "
    "Fill in the gaps of the image, which shows the object in <image_input> from a different camera viewpoint "
    "by applying the relative camera transformation (theta sin(phi) cos(phi) r)",

    "The camera viewpoint is defined by the polar angle theta, "
    "the azimuth angle phi and the radius r, with the change in viewpoint represented using their differences. "
    "Finish the incomplete image of the reference object in <image_input> which shows a novel view "
    "after applying the camera transformation (theta sin(phi) cos(phi) r)",

    "The camera perspective is determined by the polar angle theta, the azimuth angle phi, and the radius r, "
    "with the change in viewpoint expressed through their respective differences. "
    "Create a complete image of the reference object in <image_input> after implementing the camera transformation (theta sin(phi) cos(phi) r)",

    "The camera's viewpoint is characterized by the polar angle theta, azimuth angle phi, and radial distance r, "
    "while the shift in perspective is denoted by their differences. "
    "Finish an image of the <image_input> by applying the camera transformation (theta sin(phi) cos(phi) r)",

    "The camera position is established by the polar angle theta, the azimuth angle phi, and the radius r, "
    "and the alteration in viewpoint is indicated by the differences between these values. "
    "Complete a new image of the <image_input> following the implementation of the camera transformation (theta sin(phi) cos(phi) r)",

    "The camera's orientation is dictated by the polar angle theta, the azimuth angle phi, and the distance r, "
    "with the modification in perspective signified by their corresponding differences. "
    "Add the missing elements to the image which shows the object in the <image_input> "
    "after executing the camera transformation (theta sin(phi) cos(phi) r)",
]


PROMPT_DICT['Prompt_Pretrain_View_Generation_Simplified']['manual'] = [
    "Relative camera transformation is in the format of {TYPE}. What is the image after applying the transformation {TRANSFORM} to <image_input>?",
    "Given transformation {TYPE}, generate the image if applying {TRANSFORM} to <image_input>.",
    "Add missing details to a novel view of the object in <image_input> by applying the relative camera transformation {TYPE}: {TRANSFORM}.",
    "Complete the image, it shows a new viewpoint of the object in <image_input> obtained by utilizing the corresponding "     # continued
    "relative camera transformation {TYPE}: {TRANSFORM}.",
    "Complete a distinct view of the object in <image_input> by executing the relative camera transformation {TYPE}: {TRANSFORM}.",
    "The <image_input> shows an object, generate the complete image of the object from a distinct camera viewpoint "    # continued
    "through the relative camera transformation {TYPE}: {TRANSFORM}.",
    "Create a complete image of the reference object in <image_input> after implementing the camera transformation {TYPE}: {TRANSFORM}.",
    "Finish an image of the <image_input> by applying the camera transformation {TYPE}: {TRANSFORM}.",
    "Complete a new image of the <image_input> following the implementation of the camera transformation {TYPE}: {TRANSFORM}.",
]

PROMPT_DICT['Prompt_Pretrain_View_Generation_Simplified_In_Context']['manual'] = [
    "Relative camera transformation is in the format of {TYPE}. For example, {CONTEXT} "   # continued
    "What is the image after applying the transformation {TRANSFORM} to <image_input>?",
    "{CONTEXT} Apply {TRANSFORM} to <image_input> is ",    # follow the context format, w/o specifying the type
    'Given transformation {TYPE} and example "{CONTEXT}", generate the image if applying "{TRANSFORM}" to <image_input>.',
    "The camera viewpoint in a spherical coordinate system is defined by {TYPE}, where {CONTEXT} Image that applying {TRANSFORM} to <image_input>: ",
    "Relative transformation format: {TYPE}. Example: {CONTEXT} Instruction: apply {TRANSFORM} to <image_input>.",
    "Add missing details to a novel view of the object in <image_input> by applying the relative camera transformation {TYPE}: {TRANSFORM}, "   # continued
    "considering the example: {CONTEXT}",
    "A new viewpoint of the object in a reference image can be obtained by utilizing the corresponding "    # continued
    "relative camera transformation {TYPE}. For instance, {CONTEXT} Now apply {TRANSFORM} to <image_input>.",
    "Complete a distinct view of the object in <image_input> by executing the relative camera transformation {TYPE}: {TRANSFORM}. "     # continued
    "Use the example for reference: {CONTEXT}",
    "The <image_input> shows an object, generate the complete image of the object from a distinct camera viewpoint "    # continued
    "through the relative camera transformation {TYPE}: {TRANSFORM}, where {CONTEXT}",
    "Note that {CONTEXT} Create a complete image of the reference object in <image_input> "    # continued
    "after implementing the camera transformation {TYPE}: {TRANSFORM}.",
]

PROMPT_DICT['Prompt_Pretrain_View_Prediction_Simplified']['manual'] = [
    "Relative camera transformation is in the format of {TYPE}. "    # continued
    "What is the transform applied to <image_input> from {REF}?",
    "The transform applied from {REF} to <image_input> is ",
    'Given transformation in the form of {TYPE}, predict the transform if applying to {REF} is <image_input>.',
    "Relative transformation format: {TYPE}. Instruction: What transform is applied to <image_input> from {REF}?",
    "The distinct view of the object in <image_input> is generated by executing the relative camera transformation {TYPE} from {REF}: ",
]


PROMPT_DICT['Prompt_Pretrain_View_Prediction_Simplified_In_Context']['manual'] = [
    "Relative camera transformation is in the format of {TYPE}. For example, {CONTEXT} "   # continued
    "What is the transform applied to <image_input> from {REF}?",
    "{CONTEXT} The transform applied from {REF} to <image_input> is ",
    'Given transformation in the form of {TYPE} and example "{CONTEXT}", predict the transform if applying to {REF} is <image_input>.',
    "Relative transformation format: {TYPE}. Example: {CONTEXT} Instruction: What transform is applied to <image_input> from {REF}?",
    "Use the example for reference: {CONTEXT} The distinct view of the object in <image_input> is generated from {REF} by executing the relative camera transformation {TYPE}: ",
]


PROMPT_DICT['audio_separation']['original'] = [
    'Separate the sound of {TARGET} in the image from the audio mixture of {TARGET} and the sound of the video.',
    'The input sound mixture is generated from mixing sounds from the image and the video. Separate the sound of the image.',
]

PROMPT_DICT['audio_separation_modality_token']['original'] = [
    'Separate the sound of {TARGET} in <image_input> from the audio mixture of {TARGET} and the sound of {VIDEO}.',
    'The input sound mixture is generated from mixing sounds from <image_input> and {VIDEO}. Separate the sound of <image_input>.',
]

PROMPT_DICT['Segment_Anything_Bbox_Color']['original'] = [
    'Segment the region in this bounding box with the corresponding color: {}',
]

PROMPT_DICT['Segment_Anything_Point_Color']['original'] = [
    'Segment the region at each point with the corresponding color: {}',
]

PROMPT_DICT['Segment_Anything_Bbox']['original'] = [
    'Segment the region in the bounding box with the format "y1 x1 y2 x2": {}',
]

PROMPT_DICT['Segment_Anything_Point']['original'] = [
    'Segment the region at each point with the format "y x": {}',
]

PROMPT_DICT['looking_at_me']['original'] = [
    'Is the person in the image looking at the camera?',
    'Is the visible face in the image looking at the camera wearer?',
]

PROMPT_DICT['talking_to_me']['original'] = [
    'Is the person in the scene talking to the camera wearer?',
    'Identify whether the person in the video is talking to the camera wearer. Answer yes or no.',
]



Image_Generation_No_Text = [
    "An image of a random picture.",
]