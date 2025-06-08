# Streamlit App Deployment

El link del deploy lo pueden encontrar en: https://uznarnia.streamlit.app/, sin embargo, es muy lenta para el propósito de nuestra app por ello recomendamos los siguientes pasos.

## Requisitos

- Python 3.10 o superior
- pip instalado

---

## Instalación y ejecución

### 1. Clona el repositorio (si aún no lo hiciste)
```bash
git clone https://github.com/marcelino123455/correct_deploy.git
```

### 2. Crea un entorno virtual

```bash
python3 -m venv venv
```

### 3. Activa el entorno virtual

```bash
source venv/bin/activate
```

> En Windows (CMD):
> ```bash
> venv\Scripts\activate
> ```

### 4. Instala las dependencias

```bash
pip install -r requirements.txt
```

### 5. Ejecuta la aplicación

```bash
streamlit run visualizador.py
```

---
### 6. Consideraciones
- No es necesario crear el enviroment, pero puede que existan cruce entre las dependecias por eso se recomienda el enviroment.
- El deploy se puede encontrar en el siguiente link https://uznarnia.streamlit.app/, sin embargo la idea de la plataforma es ver como los parámetros varían lo cual recalcula el proceso de k-means y hace que la página del deploy sea muy lenta, por ende, poder visualizar con dinamismo la aplicación recomendamos encarecidamente correrlo de manera local.

