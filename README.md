# Projet 05 — Risques opérationnels et évaluation de la mourabaha

Application académique développée en Python avec Flask. Elle regroupe quatre modules : scoring de crédit, analyse de contrats, évaluation des fournisseurs et détection d’indicateurs de blanchiment de capitaux.

## Fonctionnalités

### 1. Scoring de crédit

Le module utilise un modèle préentraîné fourni dans `xgboost_mourabaha_final.pkl`.

À partir de 14 données saisies, l’application calcule six indicateurs supplémentaires : DTI, LTV, EAD, couverture des garanties, ratio EAD et rapport entre mensualité et financement.

Des règles de refus automatique vérifient notamment :

- L’âge du client à la fin du contrat.
- La part du salaire consacrée à la mensualité.
- Le rapport entre financement et garantie.
- La capacité des mensualités à couvrir le principal sur la durée prévue.

En dehors de ces refus, le score issu du modèle reçoit des ajustements prédéfinis liés à l’âge et à l’ancienneté professionnelle. Ces ajustements constituent des règles expertes, et non une calibration statistique validée.

### 2. Analyse de contrats

Le module extrait le texte des fichiers PDF, TXT et DOCX. Avec une clé Gemini configurée, il recherche :

- Les dates critiques.
- Les parties au contrat.
- Les clauses pénales.
- Les clauses présentant un risque.
- Les contradictions avec la politique illustrative définie dans le code.

Les PDF constitués uniquement d’images ne sont pas pris en charge par un dispositif OCR.

### 3. Évaluation des fournisseurs

Le module combine le taux de retard déclaré et des informations textuelles pour produire des signaux d’alerte, une appréciation des textes, un score de risque et une recommandation.

### 4. Analyse des transactions — AML

Le module utilise Gemini pour extraire 15 variables d’un rapport textuel, puis transmet ces variables au modèle fourni dans `aml_xgboost_model.pkl`.

Il affiche les informations extraites, le score de risque et la décision du modèle.

## Modes de démonstration

Sans clé Gemini :

- L’analyse des contrats renvoie une réponse illustrative prédéfinie.
- L’évaluation des fournisseurs utilise des règles simplifiées.
- L’extraction AML renvoie des valeurs prédéfinies indépendantes du contenu du rapport.

Si le modèle AML ne peut pas être chargé, l’application utilise une logique de simulation. Ces résultats ne constituent pas une analyse effective par les modèles correspondants.

## Fichiers

- `app.py` : serveur Flask et fonctions d’évaluation.
- `nlp_engine.py` : extraction de texte et intégration de Gemini.
- `xgboost_mourabaha_final.pkl` : modèle préentraîné de scoring.
- `aml_xgboost_model.pkl` : modèle préentraîné AML.
- `templates/index.html` : interface du navigateur.
- `static/style.css` : présentation de l’interface.
- `requirements.txt` : dépendances.
- `start.bat` : script de lancement sous Windows.
- `.env.example` : exemple de configuration sans clé secrète.

Les scripts d’entraînement des modèles ne sont pas inclus dans cette version.

## Installation

Depuis le dossier du projet :

```bash
python -m pip install -r requirements.txt
```

Conserver les versions indiquées dans `requirements.txt` pour limiter les incompatibilités avec les modèles préentraînés.

## Configuration de Gemini

Pour activer les fonctions Gemini, créer un fichier `.env` dans le même dossier que `app.py` et y renseigner :

```dotenv
GEMINI_API_KEY=votre_cle_personnelle
```

Ne jamais publier ce fichier ni une véritable clé API.

Avec Gemini activé, les textes soumis aux fonctions d’analyse sont transmis au service externe.

## Lancement local

```bash
python app.py
```

Ouvrir ensuite :

http://127.0.0.1:5000

Sous Windows, le fichier `start.bat` propose également l’installation des dépendances et le lancement.

## Portée

Le projet constitue une démonstration académique. Les règles, scores et politiques intégrés nécessitent une validation avant toute utilisation opérationnelle. Les résultats doivent faire l’objet d’une vérification humaine.

Le serveur est configuré en mode développement ; cette configuration doit être adaptée avant un déploiement public.
