import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0"}

# Use any chapter URL from the site, e.g. Genesis chapter 1
url = "https://www.vatican.va/archive/ENG0839/__P3.HTM"
html = requests.get(url, headers=HEADERS).text
soup = BeautifulSoup(html, "html.parser")

# Show the raw body structure
print(soup.body.prettify()[:3000])