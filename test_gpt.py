# pip install happytransformer
# doc : https://happytransformer.com/text-generation/settings/?ref=vennify.ai
# trouver des models entrainé : https://huggingface.co/
########################################################### pour generer la suite d un texte #############################################
from happytransformer import HappyGeneration
from happytransformer import GENSettings

happy_gen = HappyGeneration("GPT-NEO", "EleutherAI/gpt-neo-1.3B") # attension il faut au moins 12GB de libre sur la ram

args = GENSettings(no_repeat_ngram_size=2, do_sample=True, early_stopping=False, top_k=50, temperature=0.9,max_length=200)

result = happy_gen.generate_text("Artificial intelligence will ", args=args)

print(result.text)

######################

happy_gen = happy_gen = HappyGeneration("pythia-1.3B-deduped", "PygmalionAI/pygmalion-1.3b") # attension il faut au moins 12GB de libre sur la ram

args = GENSettings(no_repeat_ngram_size=2, do_sample=True, early_stopping=False, top_k=10, temperature=0.9,max_length=30)
gg = """"
LYOKO's Persona: she is very kind and polite
<START>
LYOKO: hello how are you
You: I am fine, thank you
LYOKO: My name is LYOKO and you?
You: My name is Pierre
LYOKO:Thank you I guess
You: For what?
LYOKO:"""
result = happy_gen.generate_text(gg, args=args)

print(result.text)

############################################################ pour repondre a des questions ##############################################
from happytransformer import HappyQuestionAnswering

happy_qa_albert = HappyQuestionAnswering("ALBERT", "mfeb/albert-xxlarge-v2-squad2")

result = happy_qa_albert.answer_question("Today's date is January 10th, 2021", "What is the date?",5)

for i in range(5):
    print(result[i])

############################################################ pour generer un texte a partir d un texte ###################################
from happytransformer import HappyTextToText, TTSettings

happy_tt = HappyTextToText("T5", "t5-base")

top_p_sampling_settings = TTSettings(do_sample=True, top_k=0, top_p=0.8, temperature=0.7,  min_length=20, max_length=20, early_stopping=True)

result = happy_tt.generate_text("translate English to French: nlp is a field of artificial intelligence", args=top_p_sampling_settings)

print(result)
print(result.text)
