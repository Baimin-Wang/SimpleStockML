import requests
from bs4 import BeautifulSoup

# Step 1: Fetch the HTML content of the Wikipedia page
url = 'https://en.wikipedia.org/wiki/Russell_1000_Index#Components'
response = requests.get(url)
response.raise_for_status()  # Ensure the request was successful

# Step 2: Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Step 3: Locate the table containing the list of companies
table = soup.find('table', {'class': 'wikitable', 'id' : 'constituents'})

# Step 4: Extract the Symbols (assuming it's the first column)
symbols = []
for row in table.find_all('tr')[1:]:  # Skip the header row
    cols = row.find_all('td')
    if len(cols) > 1:  # Ensure there are enough columns
        symbol = cols[1].text.strip()  # Adjust index based on inspection
        if symbol:  # Avoid empty symbols
            symbols.append(symbol)


# Step 5: Print or save the symbols
print(f"Extracted {len(symbols)} symbols:")
print(symbols)

# Optionally, save to a file
with open('russell_1000_symbols.txt', 'w') as file:
    file.write('\n'.join(symbols))