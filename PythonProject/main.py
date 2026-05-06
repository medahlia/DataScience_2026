import requests
import time



BASE_URL = "https://public-api.nazk.gov.ua/v2"


def search_declarations(name, page=1):
    url = f"{BASE_URL}/documents/list"

    params = {
        "q": name,
        "page": page
    }

    response = requests.get(url)
    return response.json()


def get_declaration(doc_id):
    url = f"https://public-api.nazk.gov.ua/v2/documents/{doc_id}"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"❌ Помилка запиту: {e}")
        return None


def check_declaration(declaration, target_child, city_filter, region_filter):
    data = declaration.get("data", {})

    # 🔹 Перевірка міста
    step_6 = data.get("step_6", {})
    city_match = False

    for item in step_6.get("data", []):
        city = (item.get("city_txt") or "").lower()
        region = (item.get("region_txt") or "").lower()

        if city_filter in city and region_filter in region:
            city_match = True
            break

    if not city_match:
        return False

    # 🔹 Перевірка дітей
    step_2 = data.get("step_2", {})

    for person in step_2.get("data", []):
        if person.get("subjectRelation") == "Дитина":
            child_name = (person.get("full_name") or "").lower()

            if target_child in child_name:
                return True

    return False


def main():
    region = input("Область: ").lower()
    city = input("Місто: ").lower()
    declarant_name = input("ПІБ (без по батькові): ").lower()
    child_name = input("ПІБ дитини: ").lower()

    page = 1
    found = False

    while page <= 10:  # обмеження, щоб не грузити API
        print(f"Перевіряю сторінку {page}...")

        result = search_declarations(declarant_name, page)
        docs = result.get("data", [])

        if not docs:
            break

        for doc in docs:
            doc_id = doc.get("id")
            declaration = get_declaration(doc_id)

            if check_declaration(declaration, child_name, city, region):
                print("\n✅ ЗНАЙДЕНО!")
                print("ID декларації:", doc_id)
                print("Посилання:", f"https://public.nazk.gov.ua/documents/{doc_id}")
                found = True
                return

            time.sleep(0.3)  # щоб не словити ліміт

        page += 1

    if not found:
        print("\n❌ Нічого не знайдено")


if __name__ == "__main__":
    main()