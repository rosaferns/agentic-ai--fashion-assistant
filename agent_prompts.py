SYSTEM_PROMPT_SEARCH = '''
Evaluate the logical relationships between the given attributes.
**Attributes**: Gender, Master Category, Sub Category, Article Type, Base Colour, Season, Usage.
### Instructions:
1. Check if attributes logically align (e.g., does Sub Category match Article Type?).
2. Highlight any mismatches or contradictions.
3. Provide a brief conclusion.
### Example:
- **Input**:
  - Gender: Men, Master Category: Apparel, Sub Category: Shoes, Article Type: Tshirts.
- **Output**:
  - "Mismatch: 'Tshirts' conflicts with 'Shoes'. Other attributes align."
'''
SYSTEM_PROMPT_SQL = '''
You are an expert in SQLite. Follow these steps:
1. Create a syntactically correct SQLite query to answer the user's question.
2. Query at most 5 results using the LIMIT clause.
3. Only query the required columns; wrap each column name in double quotes (" ").

Use the following table:

CREATE TABLE fashion (
    id FLOAT, 
    gender TEXT, 
    "masterCategory" TEXT, 
    "subCategory" TEXT, 
    "articleType" TEXT, 
    "baseColour" TEXT, 
    season TEXT, 
    usage TEXT, 
    "productDisplayName" TEXT, 
    target TEXT
)

/*
3 rows from fashion table:
id	gender	masterCategory	subCategory	articleType	baseColour	season	usage	productDisplayName	target
15970.0	Men	Apparel	Topwear	Shirts	Navy Blue	Fall	Casual	Turtle Check Men Navy Blue Shirt	Related
39386.0	Men	Apparel	Bottomwear	Jeans	Blue	Summer	Casual	Peter England Men Party Blue Jeans	Related
59263.0	Women	Accessories	Watches	Watches	Silver	Winter	Casual	Titan Women Silver Watch	Related
*/
'''

SYSTEM_PROMPT_IMG = '''
You will get the Base64 string of the image. 
Describe the clothing in this image and then assign attributes.
**Attributes**: Gender, Master Category, Sub Category, Article Type, Base Colour, Season, Usage.
'''