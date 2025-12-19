from typing import List, Set
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def extract_reviews_from_soup(soup: BeautifulSoup) -> List[str]:
    texts: Set[str] = set()

    # 1) По itemprop (если Lamoda это использует)
    for block in soup.find_all(attrs={"itemprop": "review"}):
        body = block.find(attrs={"itemprop": "reviewBody"})
        if body:
            t = body.get_text(strip=True)
            if t:
                texts.add(t)

    # 2) Если itemprop нет — пробуем типичные классы для блока текста отзыва
    # сюда легко добавить свои, когда увидишь реальную верстку
    possible_selectors = [
        ".ReviewCard__text",
        ".reviews-list__text",
        ".product-review__text",
        "[data-review-text]",           # если хранят текст в data-атрибуте
        "[data-test-id*='review']",     # часто тестовые id содержат слово review
    ]

    for sel in possible_selectors:
        for tag in soup.select(sel):
            t = tag.get_text(strip=True)
            if t:
                texts.add(t)

    # 3) На всякий случай: ищем элементы с itemtype=Review, если они есть
    for tag in soup.find_all(attrs={"itemtype": lambda v: v and "Review" in v}):
        t = tag.get_text(" ", strip=True)
        if t and len(t.split()) > 3:
            texts.add(t)

    return list(texts)


def get_lamoda_reviews_selenium(url: str, timeout: int = 20, headless: bool = False) -> List[str]:
    options = Options()

    # headless делаем параметром — сначала лучше запускать с окном, чтобы посмотреть DOM
    if headless:
        options.add_argument("--headless=new")

    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    options.add_argument("--lang=ru-RU")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)

        # Часто отзывы на отдельной вкладке — можно кликнуть по табу "Отзывы", если он есть
        try:
            reviews_tab = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located(
                    (By.XPATH, "//button[contains(., 'Отзывы') or contains(., 'отзывы')]")
                )
            )
            try:
                reviews_tab.click()
            except Exception:
                pass
        except Exception:
            # если таба нет/не нашёлся — просто идём дальше
            pass

        # Скроллим вниз, чтобы догрузились ленивые блоки
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)

        # Ждём появления блока, который потенциально содержит отзыв
        try:
            WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located(
                    (
                        By.CSS_SELECTOR,
                        "[itemprop='review'], .ReviewCard__text, .reviews-list__text, .product-review__text",
                    )
                )
            )
        except Exception:
            # если не дождались — всё равно пробуем парсить то, что есть
            pass

        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        reviews = extract_reviews_from_soup(soup)
        return reviews

    finally:
        driver.quit()


if __name__ == "__main__":
    url = "https://www.lamoda.ru/p/mp002xw0vwfx/shoes-abricot-valenki/"
    reviews = get_lamoda_reviews_selenium(url, headless=False)
    print(f"Найдено отзывов: {len(reviews)}")
    for i, r in enumerate(reviews, 1):
        print(f"\n--- Отзыв #{i} ---\n{r}")