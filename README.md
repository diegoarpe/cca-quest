# ⚔️ Architect Quest — CCA Study Tracker

Tracker gamificado para preparar la certificación **Claude Certified Architect (CCA)** de Anthropic.

Plan de estudio de 8 semanas con quizzes, challenges prácticos, sistema de XP y progresión por niveles.

## Demo

**https://diegoarpe.github.io/cca-quest/**

## Qué incluye

- **Quiz diagnóstico** de 20 preguntas que evalúa tu nivel en 5 dominios
- **8 semanas** de estudio con tareas, recursos y challenges
- **Sistema de XP y niveles** — ganá XP completando tareas (50), challenges (200), pomodoros (25) y achievements (100)
- **Progresión bloqueada** — cada semana se desbloquea al completar el challenge de la anterior
- **5 dominios**: Agentic Architecture, MCP Integration, Claude Code, Prompt Engineering, Context Management
- **Pomodoro timer** integrado
- **Notas** por dominio
- **Sync entre dispositivos** vía GitHub Gist
- **Export/Import JSON** como fallback offline

## Dominios del examen

| Dominio | Peso |
|---|---|
| Agentic Architecture | 27% |
| MCP Integration | 19% |
| Claude Code | 19% |
| Prompt Engineering | 20% |
| Context Management | 15% |

## Setup rápido

### Opción 1: Usar directamente (recomendado)

Abrí **https://diegoarpe.github.io/cca-quest/** en tu browser. Listo, tu progreso se guarda en el `localStorage` de tu browser.

### Opción 2: Hostearlo vos

1. Hacé fork de este repo
2. En tu fork: **Settings → Pages → Source: Deploy from branch → main / root → Save**
3. En ~1 minuto tenés tu propia URL: `https://TU_USUARIO.github.io/cca-quest/`

### Opción 3: Local

Descargá `index.html` y abrilo en tu browser. Funciona sin servidor.

## Sync entre dispositivos

El progreso vive en `localStorage` (aislado por browser/dispositivo). Para sincronizar entre dispositivos (ej: Mac y iPhone):

### Setup (una sola vez)

1. Ir a [github.com/settings/tokens](https://github.com/settings/tokens) → **Generate new token (classic)**
2. Nombre: `cca-quest-sync` — Scope: solo **`gist`** — Generar
3. Copiar el token (`ghp_...`)

### Uso diario

1. **Dashboard → Settings → ☁️ Sync** → pegar el token
2. **Push** al terminar de estudiar (sube tu progreso a un Gist privado en tu cuenta)
3. **Pull** desde otro dispositivo (descarga el progreso del Gist)

Cada persona usa su propio token → su propio Gist. Los datos nunca se mezclan entre usuarios.

### Fallback: Export/Import

Si no querés usar GitHub:
- **Export JSON** → copia tu progreso al clipboard
- **Import JSON** → pegá un JSON para restaurar el estado

## Acceso desde iPhone

1. Abrí la URL en **Safari**
2. Tocá el botón de **compartir** (cuadrado con flecha)
3. **"Agregar a pantalla de inicio"**
4. Se instala como una web app

## Estructura del XP

| Acción | XP |
|---|---|
| Completar tarea | +50 |
| Completar challenge semanal | +200 |
| Achievement desbloqueado | +100 |
| Pomodoro completado | +25 |
| Dominio 100% completado | +300 bonus |

El XP se recalcula desde el estado actual. Si desmarcás una tarea, perdés los XP correspondientes.

## Niveles

| Nivel | Título | XP requerido |
|---|---|---|
| 1 | Novice | 0 |
| 2 | Apprentice | 500 |
| 3 | Practitioner | 1,200 |
| 4 | Engineer | 2,000 |
| 5 | Senior Engineer | 3,000 |
| 6 | Solutions Architect | 4,200 |
| 7 | Principal Architect | 5,500 |
| 8 | Claude Certified Architect | 7,000 |

## Recursos de estudio

- [Claude 101](https://anthropic.skilljar.com/claude-101)
- [Building with the Claude API](https://anthropic.skilljar.com/building-with-the-claude-api)
- [Intro to MCP](https://anthropic.skilljar.com/intro-to-model-context-protocol)
- [Claude Code Developer Training](https://anthropic.skilljar.com/claude-code-developer-training)
- [Anthropic Docs](https://docs.anthropic.com)
- [MCP Specification](https://modelcontextprotocol.io)
- [CCA Practice Exams](https://claudecertifications.com)

## Tech

Un solo archivo HTML (~1450 líneas). Sin dependencias, sin framework, sin build step. Vanilla JS + CSS + localStorage.

## Licencia

Uso libre. Hacé fork y adaptalo a tu gusto.
