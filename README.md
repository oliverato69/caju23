---
title: caju23
emoji: 🐳
colorFrom: purple
colorTo: yellow
sdk: static
pinned: false
tags:
  - deepsite
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Robot de trading para MT5 con IA

Se agregó el archivo `MT5_AI_Trading_Robot.mq5`, un Expert Advisor para MetaTrader 5 con:

- Modelo de IA online (regresión logística) que aprende con cada señal.
- Features técnicas: EMA rápida/lenta, RSI, ADX, ATR y momentum.
- Gestión de riesgo: lotaje por porcentaje, SL/TP por ATR, filtro de spread y corte de pérdida diaria.
- Filtro horario y límite de posiciones abiertas por símbolo/magic number.

### Cómo usarlo en MT5

1. Copia `MT5_AI_Trading_Robot.mq5` a:
   - `MQL5/Experts/`
2. Abre MetaEditor y compílalo.
3. En MT5, arrástralo al gráfico del símbolo que quieres operar.
4. Ajusta parámetros clave:
   - `RiskPercent`
   - `BuyThreshold` / `SellThreshold`
   - `StopATRMultiplier` / `TakeATRMultiplier`
   - `WorkTF`
5. Haz backtest en el Strategy Tester antes de usar cuenta real.

### Recomendación importante

No existe una “mejor estrategia” universal ni precisión perfecta.
Este EA está diseñado como base sólida para iterar, validar y optimizar por activo/mercado.
