from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import pipeline
import torch


class TransformArticle:
    def __init__(self, text_en, text_ru=""):
        self.text_ru = text_ru
        self.text_en = text_en

    def translate_text(self):
        """This function translates English text to Russian."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = "utrobinmv/t5_translate_en_ru_zh_small_1024"
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model.to(device)
        tokenizer = T5Tokenizer.from_pretrained(model_name)

        prefix = "translate to ru: "
        src_text = prefix + self.text_en

        input_ids = tokenizer(src_text, return_tensors="pt")
        generated_tokens = model.generate(**input_ids.to(device))
        result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return " ".join(result)

    def emotion(self):
        """This function defines emotions."""
        pipe = pipeline(
            "text-classification",
            model="MaxKazak/ruBert-base-russian-emotion-detection",
        )

        translate_emotion = {
            "joy": "Радостная",
            "interest": "Интересная",
            "surprise": "Сюрприз",
            "sadness": "Грусть",
            "anger": "Гнев",
            "disgust": "Отвращение",
            "fear": "Страх",
            "guilt": "Вина",
            "neutral": "Нейтральная",
            "average": "Нейтральная",
        }

        result = pipe(self.text_ru)

        return translate_emotion[result[0]["label"]]

    def category(self):
        """This function defines category."""
        pipe = pipeline(
            "text-classification",
            model="akashmaggon/bert-base-uncased-newscategoryclassification-fullmodel-5",
        )

        translate_category = {
            "ARTS": "ИСКУССТВО",
            "ARTS & CULTURE": "ИСКУССТВО И КУЛЬТУРА",
            "BLACK VOICES": "ЧЕРНЫЕ ГОЛОСА",
            "BUSINESS": "БИЗНЕС",
            "COLLEGE": "Колледж",
            "COMEDY": "КОМЕДИЯ",
            "CRIME": "ПРЕСТУПНОСТЬ",
            "CULTURE & ARTS": "КУЛЬТУРА И ИСКУССТВА",
            "DIVORCE": "РАЗВОД",
            "EDUCATION": "ОБРАЗОВАНИЕ",
            "ENTERTAINMENT": "РАЗВЛЕЧЕНИЕ",
            "ENVIRONMENT": "ОКРУЖАЮЩАЯ СРЕДА",
            "FIFTY": "ПЯТЬДЕСЯТ",
            "FOOD & DRINK": "ПИТАНИЕ И НАПИТКИ",
            "GOOD NEWS": "ХОРОШИЕ НОВОСТИ",
            "GREEN": "ЗЕЛЕНЫЙ",
            "HEALTHY LIVING": "ЗДОРОВОЕ ЖИЗНЬ",
            "HOME & LIVING": "ДОМ И ЖИЗНЬ",
            "IMPACT": "ВЛИЯНИЕ",
            "LATINO VOICES": "ЛАТИНО ГОЛОСА",
            "MEDIA": "СМИ",
            "MONEY": "ДЕНЬГИ",
            "PARENTING": "РОДИТЕЛИ",
            "PARENTS": "РОДИТЕЛИ",
            "POLITICS": "ПОЛИТИКА",
            "QUEER VOICES": "ГОЛОСЫ ЛГБТК",
            "RELIGION": "РЕЛИГИЯ",
            "SCIENCE": "НАУКА",
            "SPORTS": "СПОРТ",
            "STYLE": "СТИЛЬ",
            "STYLE & BEAUTY": "СТИЛЬ И КРАСОТА",
            "TASTE": "ВКУС",
            "TECH": "ТЕХНОЛОГИИ",
            "THE WORLDPOST": "МИРОВЫЕ НОВОСТИ",
            "TRAVEL": "ПУТЕШЕСТВИЯ",
            "U.S. NEWS": "НОВОСТИ США",
            "WEDDINGS": "СВАДЬБЫ",
            "WEIRD NEWS": "СТРАННЫЕ НОВОСТИ",
            "WELLNESS": "ЗДОРОВЬЕ",
            "WOMEN": "ЖЕНЩИНЫ",
            "WORLD NEWS": "МИРОВЫЕ НОВОСТИ",
            "WORLDPOST": "МИРОВЫЕ ПОСТЫ",
        }

        result = pipe(self.text_en)

        return translate_category[result[0]["label"]]

    def total_return(self):
        translate_text = self.translate_text()
        emotion = self.emotion()
        category = self.category()

        return translate_text, emotion, category
