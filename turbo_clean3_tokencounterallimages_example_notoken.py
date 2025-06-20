import openai
import os
import base64
import pandas as pd
import json
import time
import tiktoken

TOTAL_COST = 0.0
MAX_COST = 200.0
ENCODING = tiktoken.encoding_for_model("gpt-4-turbo-2024-04-09")  # adjust if you change model

# 🔐 Set your API key
openai.api_key = ""  # Replace with your OpenAI key

# 📁 Folder with your 3900 images
TRAINING_IMAGE_DIR = r"I:\turbo\adairtrain2"
BATCH_IMAGE_DIR = r"I:\turbo\all images"

OUTPUT_CSV = r"I:\turbo\extracted_200_clean2.csv"


HEADERS = [
    "Soil Series", "County", "State", "Site No.", "Texture Modifier", "Surface Texture", "Particle Size", "Mineralogy",
    "Temperature", "Reaction", "Other", "Sub Group", "Great Group",
    "Location", "Section", "Township", "Range", "Physiography", "Elevation", "Topography",
    "Slope", "Aspect", "Drainage", "Vegetation", "Date Described", "1", "2", "3", "Depth to carbonates", "Site Remarks"
]

# 👇 Add your image examples and expected output
EXAMPLES = [
    {
        "filename": "Adair Site 9.jpg",
        "data": {
            "Soil Series": "Sicl", "County": "Adair", "State": "Iowa", "Site No.": "1-192D1-1",
            "Texture Modifier" : "gravelly", "Surface Texture": "clay loam", "Particle Size": "fine-loamy",
            "Mineralogy" : "mixed", "Temperature" : "mesic", "Reaction": "calcareous", "Other" : "[empty]",
            "Sub Group" : "Aquic", "Great Group" : "Hapludolls",
            "Location": "370ft W of fence and 114ft S of fence", "Section": "2", "Township": "77N",
            "Range": "30W", "Physiography": "upland", "Elevation": "960ft.", "Topography": "strongly sloping",
            "Slope": "9-14", "Aspect": "E", "Drainage": "moderately well",
            "Vegetation": "pasture", "Date Described": "Wed, Jun 2, 1971", "1": "Glacial Till", "2": "Loess", "3": "Alluvium",
            "Depth to carbonates" : "[empty]",
            "Site Remarks": "0.18 miles N and 400ft W of SE cor of Sec 2, 77N, R30W."
        }
    },
    {
        "filename": "Adair Site 10.jpg",
        "data": {
            "Soil Series": "Clinton", "County": "Adair", "State": "Iowa", "Site No.": "1-80-1",
            "Texture Modifier" : "[empty]", "Surface Texture": "[empty]", "Particle Size": "[empty]",
            "Mineralogy" : "[empty]", "Temperature" : "[empty]", "Reaction": "[empty]", "Other" : "[empty]",
            "Sub Group" : "[empty]", "Great Group" : "[empty]",
            "Location": "2500ft W and 2200ft S of NE cor", "Section": "35", "Township": "76N",
            "Range": "30W","Physiography": "upland", "Elevation": "[empty]", "Topography": "gently sloping",
            "Slope": "2-5", "Aspect": "N", "Drainage": "moderately well",
            "Vegetation": "full timber",
            "Date Described": "Tue, Jun 15, 1971", "1": "Loess", "2": "[empty]", "3": "[empty]","Depth to carbonates" : "[empty]",
            "Site Remarks": "[empty]"
        }
    },
    {
        "filename": "Adair Site 19.jpg",
        "data": {
            "Soil Series": "Colo", "County": "Adair", "State": "Iowa", "Site No.": "1-133-1",
            "Texture Modifier" : "[empty]", "Surface Texture": "silty clay loam", "Particle Size": "fine-silty",
            "Mineralogy" : "mixed", "Temperature" : "mesic", "Reaction": "[empty]", "Other" : "[empty]",
            "Sub Group" : "Cumulic", "Great Group" : "Haplaquolls",
            "Location": "924ft S of farmstead driveway and 228ft E of fence", "Section": "5",
            "Township": "74N", "Range": "33W",
            "Physiography": "bottomland", "Elevation": "[empty]", "Topography": "nearly level",
            "Slope": "0-2", "Aspect": "CONV", "Drainage": "poorly",
            "Vegetation": "cultivated field", "Date Described": "Wed, Sep 6, 1972",
            "1": "Alluvium", "2": "[empty]", "3": "[empty]","Depth to carbonates" : "39",
            "Site Remarks": "600ft N and 1000ft W of SE cor."
        }
    },
    {
        "filename": "Adair Site 32.jpg",
        "data": {
            "Soil Series": "Ely", "County": "Adair", "State": "Iowa", "Site No.": "1-428-3",
            "Texture Modifier" : "[empty]", "Surface Texture": "silty clay loam", "Particle Size": "fine-silty",
            "Mineralogy" : "mixed", "Temperature" : "mesic", "Reaction": "[empty]", "Other" : "[empty]",
            "Sub Group" : "Cumulic", "Great Group" : "Hapludolls",
            "Location": "400ft N and 45ft E of fence at W1/4 corner", "Section": "10",
            "Township": "75N", "Range": "33W",
            "Physiography": "bottomland", "Elevation": "[empty]", "Topography": "nearly level",
            "Slope": "0-2", "Aspect": "[empty]", "Drainage": "somewhat poorly",
            "Vegetation": "cultivated field", "Date Described": "Thu, Jul 26, 1973",
            "1": "Alluvium", "2": "[empty]", "3": "[empty]","Depth to carbonates" : "[empty]",
            "Site Remarks": "1. Location is 400ft N of road. Also 70ft N and 45ft E of power line pole that is north of drainageway. 2. East of 6th fence post north of power line pole."
        }
    },
]

# 🔧 Convert image to base64
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# 🧠 Build few-shot message list
def build_messages(target_image_path):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an OCR assistant. Extract exactly the following 29 fields from the image:\n"
                "  - Soil Series, County, State, Site No.,\n"
                "  -  Texture Modifier, Surface Texture,\n"
                "  -  Particle Size, Mineralogy,\n"
                "  -  Temperature, Reaction, Other,\n"
                "  -  Sub Group, Great Group, \n"
                "  -  Location, \n"
                "  -  Section, Township, Range, \n"
                "  -  Physiography, Elevation, Topography, \n"
                "  -  Slope, Aspect, Drainage, Vegetation, \n"
                "  -  Date Described, \n"
                "  -  1, 2, 3, \n"
                "  -  Site Remarks \n\n"
                "Only extract visible data. Do not infer or guess. Return values as a valid JSON object with only those 29 keys.\n"
                "Use '[empty]' for any field that's blank. No markdown. No extra text."
            )
        }
    ]

    for example in EXAMPLES:
        example_path = os.path.join(TRAINING_IMAGE_DIR, example["filename"])
        if os.path.exists(example_path):
            base64_img = encode_image(example_path)
            image_url = f"data:image/jpeg;base64,{base64_img}"

            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image:"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": json.dumps(example["data"])}
                ]
            })

    # Add target image
    base64_target = encode_image(target_image_path)
    target_url = f"data:image/jpeg;base64,{base64_target}"
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": "Image:"},
            {"type": "image_url", "image_url": {"url": target_url}}
        ]
    })

    return messages


    # Add target image
    base64_target = encode_image(target_image_path)
    target_url = f"data:image/jpeg;base64,{base64_target}"
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": "Image:"},
            {"type": "image_url", "image_url": {"url": target_url}}
        ]
    })

    return messages

# 🔍 Extract JSON from image
def extract_fields_from_image(image_path):
    global TOTAL_COST

    try:
        messages = build_messages(image_path)

        # Count prompt tokens
        prompt_texts = []
        for msg in messages:
            if isinstance(msg["content"], list):
                for part in msg["content"]:
                    if part["type"] == "text":
                        prompt_texts.append(part["text"])
            else:
                prompt_texts.append(msg["content"])
        joined_prompt = "\n".join(prompt_texts)
        prompt_tokens = len(ENCODING.encode(joined_prompt))

        # Make request
        response = openai.chat.completions.create(
            model="gpt-4-turbo-2024-04-09",
            messages=messages,
            temperature=0,
            max_tokens=1500
        )

        completion_text = response.choices[0].message.content.strip()
        completion_tokens = len(ENCODING.encode(completion_text))

        # Cost calculation
        cost = (prompt_tokens / 1000) * 0.01 + (completion_tokens / 1000) * 0.03
        TOTAL_COST += cost
        print(f"💰 Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, Cost: ${cost:.4f}, Total so far: ${TOTAL_COST:.2f}")

        if TOTAL_COST > MAX_COST:
            print("🛑 Stopping — Total cost exceeded $500.")
            return "STOP"

        if completion_text.startswith("```json"):
            completion_text = completion_text.replace("```json", "").replace("```", "").strip()

        return json.loads(completion_text)

    except Exception as e:
        print(f"❌ Failed on {os.path.basename(image_path)}: {e}")
        return {key: "[error]" for key in HEADERS}


# 🚀 Run full batch
def run_batch_extraction():
    image_paths = sorted([
        os.path.join(BATCH_IMAGE_DIR, f)
        for f in os.listdir(BATCH_IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    start_time = time.time()

    for i, img_path in enumerate(image_paths, 1):
        print(f"\n🔍 Processing {i}/{len(image_paths)}: {os.path.basename(img_path)}")
        t0 = time.time()

        result = extract_fields_from_image(img_path)

        if result == "STOP":
            break

        df = pd.DataFrame([[f'"{result.get(h, "[error]")}"' for h in HEADERS]], columns=HEADERS)

        if os.path.exists(OUTPUT_CSV):
            df.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
        else:
            df.to_csv(OUTPUT_CSV, mode="w", header=True, index=False)

        print(f"✅ Row saved — {round(time.time() - t0, 2)} sec")

    total_time = round(time.time() - start_time, 2)
    print(f"\n🕒 Total time: {total_time} seconds — Final estimated cost: ${TOTAL_COST:.2f}")


# 🏁 Run it
if __name__ == "__main__":
    run_batch_extraction()

