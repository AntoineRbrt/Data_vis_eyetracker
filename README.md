# Data analysis eye-tracking — V1

## Images

Placer les 6 images expérimentales (3840×2160 px) ici, avec ces noms :

```
assets/images/REF1.png
assets/images/REF2.png
assets/images/COMP1.png
assets/images/COMP2.png
assets/images/COMP3.png
assets/images/TRIO.png
```

## Lancer l'application en local

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Déployer / mettre à jour sur Streamlit Community Cloud

1. Pousser ce dossier (avec les images dans `assets/images/`) sur le dépôt GitHub :
   ```bash
   git add .
   git commit -m "V1 de l'application d'analyse eye-tracking"
   git push
   ```
2. Sur [share.streamlit.io](https://share.streamlit.io), l'application liée à ce dépôt se
   redéploie automatiquement à chaque `push` sur la branche configurée.
