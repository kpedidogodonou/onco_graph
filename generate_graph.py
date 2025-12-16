from rdflib.namespace import RDFS, XSD, RDF, SDO, FOAF
from rdflib import Literal, Namespace, Graph
import pandas as pd
import requests
import tarfile
import pathlib
import pickle
import rdflib
import os
import re
import json

# Defining main directory
DATA_DIR = "./data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CLINICAL_DIRECTORY = os.path.join(RAW_DIR, "clinical")
CLINICAL_DIR_PATH = pathlib.Path(CLINICAL_DIRECTORY)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# Various variables
DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"
TARGET_COLUMN = "diagnoses.primary_diagnosis"
ONTOLOGY = "ncit"

# Namespaces for the knowledge graph
OG = Namespace("http://www.oncograph.net/hospital-data/")
NCIT = Namespace("https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/Thesaurus_25.11d.OWL#")
SCHEMA = Namespace("https://schema.org/")

def get_ontology_code(concept: str) -> str:
    """
    Return the NCIT ontology code for a given diagnosis concept.

    Args:
        concept (str): Human-readable diagnosis name (e.g. "Melanoma, NOS").

    Returns:
        Optional[str]:
            - NCIT code in the form "NCIT:C3224" if found
            - NO_MATCH if the concept cannot be mapped
    """

    base_url = "https://www.ebi.ac.uk/ols/api/search"
    params = {
        "q": concept,
        "ontology": ONTOLOGY,
        "rows": 1,
        "exact": "false"
    }

    try:
        r = requests.get(base_url, params=params)
        data = r.json()
        docs = data.get("response", {}).get("docs", [])

        if docs:
            return docs[0].get("obo_id", "NO_MATCH")
        return "NO_MATCH"
    except Exception as e:
        print(f"ERROR: {e}")
        return "NO_MATCH"


with open(os.path.join(DATA_DIR, "manifest.txt")) as f:
    # Get the file IDs from the MANIFEST file
    print("Getting clinical file IDs...")
    clinical_file_ids = [line.split()[0]  for line in f.readlines()[1:] if "Clinical_Supplement" not  in line.split()[1] ]
    print(f" {len(clinical_file_ids)} File IDs found!")

    # Download the files using the retrieved IDs
    print("Downloading files...")
    request_params = {"ids": clinical_file_ids}
    try:
        response = requests.post(DATA_ENDPOINT, data=json.dumps(request_params), headers={"Content-Type": "application/json"})

        # Retrieve the filename located inside the Content-Disposition header
        response_head_cd = response.headers["Content-Disposition"]
        file_name  = re.findall("filename=(.+)", response_head_cd)[0]
        clinical_files_compressed = os.path.join(RAW_DIR, file_name)

        # Save the compressed file on disk
        with open(clinical_files_compressed, "wb") as f:
            f.write(response.content)
        print("Files downloaded!")

        # Extract the content
        print("Extracting clinical files...")
        clinical_files = tarfile.open(clinical_files_compressed)
        clinical_files.extractall(CLINICAL_DIRECTORY)
        clinical_files.close()
    except Exception as e:
        print(f"ERROR: {e}")
        print(f"ERROR: {clinical_files_compressed} could not be extracted!")
    

# Moving the clinical files from their respective directory to the root Clinical Directory
for file in CLINICAL_DIR_PATH.rglob("*"):
    if file.is_file():
        target = CLINICAL_DIR_PATH / file.name
        file.rename(target)

# Delete the empty folders
for folder in CLINICAL_DIR_PATH.rglob("*"):
    if folder.is_dir():
        folder.rmdir()

# Merge the clinical files into a unique file
dfs = []
clinical_files = CLINICAL_DIR_PATH.glob("FM-AD*.tsv")
for file in clinical_files:
    df = pd.read_csv(file, sep="\t")
    dfs.append(df)

merged_clinical_df = pd.concat(dfs, axis=0, ignore_index=True)
merged_clinical_df.to_csv(f"{PROCESSED_DIR}/merged_clinical_df.csv", index=False)

df['diagnoses.age_at_diagnosis'] = pd.to_numeric(df['diagnoses.age_at_diagnosis'], downcast='integer', errors='coerce')

# Extract the list of unique concepts in the column "diagnoses.primary_diagnosis"
df = pd.read_csv(f"{PROCESSED_DIR}/merged_clinical_df.csv")
unique_concepts = df[TARGET_COLUMN].value_counts().index.tolist()

print(unique_concepts)
print(len(unique_concepts))

# Create an ontology dict from the unique concept
ontology_dict={}
for concept in unique_concepts:
    # Get the NCIT identifier for each concept
    ontology_code = get_ontology_code(concept)
    print(concept, "====>", ontology_code)
    ontology_dict[concept] = ontology_code


print(ontology_dict)
try:
    # Save the ontology dictionary on disk
    ontology_dict_file = open(f"{PROCESSED_DIR}/ontology_dict.pickle", "wb")
    pickle.dump(ontology_dict, ontology_dict_file)
    ontology_dict_file.close()
except:
    print("No ontology dictionary file created")


# Use the ontology dict dictionary to create a new column "ncit_code" in the dataset
# Containing the corresponding NCIT identifier for the value in the "diagnoses.primary_diagnosis" column
with open(f"{PROCESSED_DIR}/ontology_dict.pickle", "rb") as f:
    oncology_dict = pickle.load(f)
    df["ncit_code"] = df[TARGET_COLUMN].map(lambda x: oncology_dict[x])
    df.to_csv(f"{PROCESSED_DIR}/completed_clinical_df.csv", index=False)


df = pd.read_csv(f"{PROCESSED_DIR}/completed_clinical_df.csv")

#Initialize the Graph
g = Graph()

# Create the triples in the format
# Patient -> hasDiagnosis -> NCIT:xxxx
for _, patient in df.iterrows():
    # Unique Identifier for the patient
    patient_id = patient["cases.submitter_id"]
    patient_uri = OG[patient_id]
    g.add((patient_uri, RDF.type, SCHEMA.Patient))

    # Unique Identifier for the diagnosis
    ncit_code_raw = patient["ncit_code"]
    ncit_code = ncit_code_raw.split(":")[-1]
    diagnosis_uri = NCIT[ncit_code]

    # Link the patient to their diagnosis using the hasDiagnosis relationship
    g.add((patient_uri, OG.hasDiagnosis, diagnosis_uri))

    patient_gender = patient['demographic.gender']
    g.add((patient_uri, SCHEMA.Gender, Literal(patient_gender, datatype=XSD.string)))

    patient_age = patient['diagnoses.age_at_diagnosis']
    g.add((patient_uri, OG.ageAtDiagnosisDays, Literal(patient_age, datatype=XSD.integer)))

    # Label the diagnosis identifier
    primary_diagnosis = Literal(patient["diagnoses.primary_diagnosis"], datatype=XSD.string)
    g.add((diagnosis_uri, RDFS.label, primary_diagnosis))

    disease_primary_site = str(patient['cases.primary_site'])
    g.add((patient_uri, OG.hasDiseasePrimarySite, Literal(disease_primary_site, datatype=XSD.string)))

# Create Readable namespaces
g.bind("ncit", NCIT)
g.bind("og", OG)
g.bind("schema", SCHEMA, override=True)

# Save the Knowledge Graph on disk
g.serialize(destination=os.path.join(PROCESSED_DIR, "knowledge_graph.ttl"), format="turtle")