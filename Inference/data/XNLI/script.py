import json
import jsonlines
import random

# langs = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]

# suffix = {"en": "If you have to choose one of these options, your answer would be:",
#           "ar": "إذا كان عليك اختيار أحد هذه الخيارات، فإن إجابتك ستكون:",
#           "bg": "Ако трябва да изберете една от тези опции, вашият отговор ще бъде:",
#           "de": "Wenn Sie eine dieser Optionen wählen müssten, wäre Ihre Antwort:",
#           "el": "Εάν πρέπει να επιλέξετε μία από αυτές τις επιλογές, η απάντησή σας θα ήταν:",
#           "es": "Si tuvieras que elegir una de estas opciones tu respuesta sería:",
#           "fr": "Si vous devez choisir l’une de ces options, votre réponse serait :",
#           "hi": "यदि आपको इनमें से कोई एक विकल्प चुनना हो तो आपका उत्तर होगा:",
#           "ru": "Если вам придется выбрать один из этих вариантов, ваш ответ будет следующим:",
#           "sw": "Ikiwa itabidi uchague moja ya chaguzi hizi, jibu lako litakuwa:",
#           "th": "หากคุณต้องเลือกหนึ่งในตัวเลือกเหล่านี้ คำตอบของคุณจะเป็น:",
#           "tr": "Bu seçeneklerden birini seçmek zorunda kalırsanız cevabınız şu olacaktır:",
#           "ur": "اگر آپ کو ان اختیارات میں سے کسی ایک کا انتخاب کرنا ہے، تو آپ کا جواب یہ ہوگا:",
#           "vi": "Nếu bạn phải chọn một trong những lựa chọn này, câu trả lời của bạn sẽ là:",
#           "zh": "如果你必须选择其中一个选项，你的答案是："}

# for lang in langs:
#     all = []
#     with jsonlines.open(f"/share/home/fengxiaocheng/yfye/ACL2025_2/Evaluation/datasets/XNLI/split/{lang}.json") as f:
#         for line in f:
#             s = line.split("\n\n")
#             line = s[1] + '\n\n' + s[0] + '\n\n' + s[2] + "\n\n" + suffix[lang]
#             all.append({"instruction": line})
#             # print(line)
#             # break
#     with jsonlines.open(f"/share/home/fengxiaocheng/yfye/ACL2025_2/Evaluation/datasets/XNLI/split/{lang}.json", "w") as f:
#         for line in all:
#             f.write(line)

random.seed(666)
sampled_list = random.sample(range(5010), 1000)

langs = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
for lang in langs:
    all = []
    with jsonlines.open(f"/share/home/fengxiaocheng/yfye/ACL2025_2/Evaluation/output_oo/XNLI/llama-3.1-8b/full_sft/{lang}.json") as f:
        for i, line in enumerate(f):
            if i in sampled_list:
                all.append(line)
    with jsonlines.open(f"/share/home/fengxiaocheng/yfye/ACL2025_2/Evaluation/output_o/XNLI/llama-3.1-8b/full_sft/{lang}.json", 'w') as f:
        for line in all:
            f.write(line)
    # all = []
    # with jsonlines.open(f"/share/home/fengxiaocheng/yfye/ACL2025_2/Evaluation/datasets/XNLI/ans/{lang}.json") as f:
    #     for i, line in enumerate(f):
    #         if i in sampled_list:
    #             all.append(line)
    # with jsonlines.open(f"/share/home/fengxiaocheng/yfye/ACL2025_2/Evaluation/datasets/XNLI/ans_sample/{lang}.json", 'w') as f:
    #     for line in all:
    #         f.write(line)