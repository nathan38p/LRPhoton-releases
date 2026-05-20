# DataSAXS Template

Prototype propre pour recommencer le logiciel DataSAXS avec une structure modulaire.

Contact : piaget.nathan@icloud.com

## Lancer l'application

```bash
cd DataSAXS_Template
python3 main.py
```

Sur ton Mac avec Python 3.14 :

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 main.py
```

## Installer les dépendances

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pip install -r requirements.txt
```

## Structure

```text
DataSAXS_Template/
├── main.py
├── requirements.txt
├── README.md
├── tabs/
│   ├── __init__.py
│   ├── find_centre_tab.py
│   ├── xenocs_cave_tab.py
│   └── id13_cave_tab.py
├── utils/
│   └── __init__.py
└── assets/
```

Chaque onglet est dans son propre fichier Python.
