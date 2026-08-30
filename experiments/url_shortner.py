import string
import random


class URLShortener:
    def __init__(self):
        self.url_to_code = {}
        self.code_to_url = {}

    def _generate_code(self, length=6):
        chars = string.ascii_letters + string.digits

        while True:
            code = ''.join(random.choices(chars, k=length))
            if code not in self.code_to_url:
                return code

    def shorten(self, url):
        # Return existing code if URL was already shortened
        if url in self.url_to_code:
            return self.url_to_code[url]

        code = self._generate_code()

        self.url_to_code[url] = code
        self.code_to_url[code] = url

        return f"https://short.ly/{code}"

    def resolve(self, short_url):
        code = short_url.rstrip("/").split("/")[-1]
        return self.code_to_url.get(code)


# Example
shortener = URLShortener()

short_url = shortener.shorten(
    "https://example.com/very/long/url"
)

print(short_url)

original_url = shortener.resolve(short_url)
print(original_url)