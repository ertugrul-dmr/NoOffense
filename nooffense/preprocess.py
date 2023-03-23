import re
import string


class TextProcessor:
    def __init__(self):
        pass

    def remove_punct(self, text: str) -> str:
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        return text.translate(translator)

    def remove_html(self, text: str) -> str:
        html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        return re.sub(html, '', text)

    def remove_mentions_and_hashtags(self, text: str) -> str:
        no_mentions_hashtags = ' '.join(
            [word for word in text.split() if not (word.startswith('@') or word.startswith('#'))])
        return no_mentions_hashtags

    def remove_URL(self, text: str) -> str:
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'', text)

    def remove_emoji(self, text: str) -> str:
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def remove_single_character_tokens(self, text: str) -> str:
        clean_text = re.sub(r'\b\w\b', '', text)
        return clean_text

    def remove_extra_whitespaces(self, text: str) -> str:
        clean_text = re.sub('\s+', ' ', text).strip()
        return clean_text

    def lowercase_text(self, text: str) -> str:
        return text.lower()

    def remove_digits(self, text: str) -> str:
        return re.sub(r'\d', '', text)

    def clean_text(self, text, remove_hash_ment=True, remove_html=True, remove_emoji=True, remove_url=True,
                   remove_punct=True, remove_single_chars=True, lowercase=False, remove_digits=False) -> str:

        if remove_hash_ment:
            text = self.remove_mentions_and_hashtags(text)

        if remove_html:
            text = self.remove_html(text)

        if remove_emoji:
            text = self.remove_emoji(text)

        if remove_url:
            text = self.remove_URL(text)

        if remove_punct:
            text = self.remove_punct(text)

        if remove_single_chars:
            text = self.remove_single_character_tokens(text)

        if lowercase:
            text = self.lowercase_text(text)

        if remove_digits:
            text = self.remove_digits(text)

        return self.remove_extra_whitespaces(text)
