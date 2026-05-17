# LunaX
Postgraduade thesis app

## Wymagania
- Python 3.10+
- uv
- Node.js 18+
- npm

## Backend (FastAPI)
Instalacja zaleznosci:

```bash
uv sync
```

Uruchomienie serwera (uvicorn przez main.py):

```bash
uv run python main.py
```

Testy backendu:

```bash
uv run pytest -q
```

## Frontend (React + react-leaflet)
Instalacja zaleznosci:

```bash
cd frontend
npm install
```

Tryb developerski (odpalamy w katalogu frontend):

```bash
npm run dev
```

Build produkcyjny (odpalamy w katalogu frontend):

```bash
npm run build
```


Jak uruchomić:

Backend: uv run python main.py
Frontend: cd frontend && npm run dev