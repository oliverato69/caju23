#property strict
#property version   "1.00"
#property description "EA para MT5 con modelo IA online (logistica) + filtros de riesgo"

#include <Trade/Trade.mqh>

CTrade trade;

input group "=== Riesgo ==="
input double RiskPercent            = 1.0;      // % de balance arriesgado por trade
input double MaxSpreadPoints        = 25;       // max spread permitido en points
input double MaxDailyLossPercent    = 3.0;      // corte diario si perdida acumulada supera %
input int    MaxOpenPositions       = 1;        // posiciones simultaneas

input group "=== Modelo IA ==="
input int    FastEMA                = 20;
input int    SlowEMA                = 50;
input int    RSIPeriod              = 14;
input int    ATRPeriod              = 14;
input int    ADXPeriod              = 14;
input double LearningRate           = 0.06;     // velocidad de aprendizaje online
input double BuyThreshold           = 0.58;     // prob minima para compra
input double SellThreshold          = 0.42;     // prob maxima para venta

input group "=== Ejecucion ==="
input ENUM_TIMEFRAMES WorkTF        = PERIOD_M15;
input int    StopATRMultiplier      = 2;
input int    TakeATRMultiplier      = 3;
input int    TradeStartHour         = 6;
input int    TradeEndHour           = 22;
input ulong  MagicNumber            = 230023;

// Pesos modelo logístico online (bias + 5 features)
double W[6] = {0.0, 0.2, -0.2, 0.1, 0.1, 0.15};

datetime lastBarTime = 0;
double   dayStartEquity = 0.0;
int      dayOfYear = -1;

int hFastEMA = INVALID_HANDLE;
int hSlowEMA = INVALID_HANDLE;
int hRSI     = INVALID_HANDLE;
int hATR     = INVALID_HANDLE;
int hADX     = INVALID_HANDLE;

// Ultimo ejemplo para entrenamiento online post-resultado
bool   hasLastSample = false;
double lastX[6];
double lastEntryPrice = 0.0;
int    lastDirection = 0; // 1 buy, -1 sell


double Sigmoid(double z)
{
   if(z > 35.0) return 0.999999;
   if(z < -35.0) return 0.000001;
   return 1.0 / (1.0 + MathExp(-z));
}

void NormalizeFeatures(double &x1,double &x2,double &x3,double &x4,double &x5)
{
   // Normalizaciones simples para estabilidad del entrenamiento
   x1 = MathTanH(x1 / 2.0);   // pendiente EMA
   x2 = MathTanH(x2 / 20.0);  // distancia RSI de 50
   x3 = MathTanH(x3 / 2.0);   // momentum corto
   x4 = MathTanH(x4 / 50.0);  // fuerza ADX
   x5 = MathTanH(x5 / 0.01);  // volatilidad relativa
}

double PredictProb(const double &x[])
{
   double z = 0.0;
   for(int i=0;i<6;i++) z += W[i]*x[i];
   return Sigmoid(z);
}

void OnlineTrain(const double &x[], double y)
{
   double p = PredictProb(x);
   double grad = (y - p);
   for(int i=0;i<6;i++)
      W[i] += LearningRate * grad * x[i];
}

bool IsNewBar()
{
   datetime t = iTime(_Symbol, WorkTF, 0);
   if(t == 0) return false;
   if(t != lastBarTime)
   {
      lastBarTime = t;
      return true;
   }
   return false;
}

bool TradingHour()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   return (dt.hour >= TradeStartHour && dt.hour <= TradeEndHour);
}

void RefreshDayState()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   if(dayOfYear != dt.day_of_year)
   {
      dayOfYear = dt.day_of_year;
      dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   }
}

bool DailyLossExceeded()
{
   if(dayStartEquity <= 0.0) return false;
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   double dd = ((dayStartEquity - eq) / dayStartEquity) * 100.0;
   return (dd >= MaxDailyLossPercent);
}

int CountOpenPositions()
{
   int count = 0;
   for(int i=PositionsTotal()-1; i>=0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if((string)PositionGetString(POSITION_SYMBOL) == _Symbol && (ulong)PositionGetInteger(POSITION_MAGIC) == MagicNumber)
         count++;
   }
   return count;
}

bool GetIndicators(double &emaFast0,double &emaSlow0,double &rsi0,double &atr0,double &adx0)
{
   double bf[3], bs[3], br[3], ba[3], bd[3];
   if(CopyBuffer(hFastEMA,0,0,3,bf) < 3) return false;
   if(CopyBuffer(hSlowEMA,0,0,3,bs) < 3) return false;
   if(CopyBuffer(hRSI,0,0,3,br) < 3) return false;
   if(CopyBuffer(hATR,0,0,3,ba) < 3) return false;
   if(CopyBuffer(hADX,0,0,3,bd) < 3) return false;

   emaFast0 = bf[0];
   emaSlow0 = bs[0];
   rsi0 = br[0];
   atr0 = ba[0];
   adx0 = bd[0];
   return true;
}

bool BuildFeatures(double &x[])
{
   ArrayResize(x,6);
   x[0] = 1.0; // bias

   double emaF, emaS, rsi, atr, adx;
   if(!GetIndicators(emaF,emaS,rsi,atr,adx)) return false;

   double close0 = iClose(_Symbol, WorkTF, 0);
   double close1 = iClose(_Symbol, WorkTF, 1);
   double close3 = iClose(_Symbol, WorkTF, 3);
   if(close0 == 0 || close1 == 0 || close3 == 0) return false;

   double f1 = (emaF - emaS) / _Point;
   double f2 = (rsi - 50.0);
   double f3 = ((close0 - close3) / close3) * 100.0;
   double f4 = adx;
   double f5 = atr / close0;

   NormalizeFeatures(f1,f2,f3,f4,f5);

   x[1] = f1;
   x[2] = f2;
   x[3] = f3;
   x[4] = f4;
   x[5] = f5;

   return true;
}

double ComputeLotByRisk(double stopDistancePrice)
{
   if(stopDistancePrice <= 0.0) return 0.0;

   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskMoney = balance * (RiskPercent / 100.0);

   double tickVal  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tickVal <= 0.0 || tickSize <= 0.0) return 0.0;

   double valuePerPrice = tickVal / tickSize;
   double lots = riskMoney / (stopDistancePrice * valuePerPrice);

   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double step   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   lots = MathMax(minLot, MathMin(maxLot, lots));
   lots = MathFloor(lots / step) * step;

   return NormalizeDouble(lots, 2);
}

bool SpreadOk()
{
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(ask <= 0 || bid <= 0) return false;
   double spreadPts = (ask - bid) / _Point;
   return spreadPts <= MaxSpreadPoints;
}

void TryTrainFromClosedSignal()
{
   if(!hasLastSample) return;

   // entrenamiento simple: compara precio actual vs entrada como proxy de resultado
   double closeNow = iClose(_Symbol, WorkTF, 0);
   if(closeNow <= 0.0) return;

   double pnl = (closeNow - lastEntryPrice) * lastDirection;
   double y = (pnl > 0.0 ? 1.0 : 0.0);
   OnlineTrain(lastX, y);
   hasLastSample = false;
}

void OpenTrade(int direction, const double &x[])
{
   double atr = 0.0;
   double tmp1,tmp2,tmp3,tmp5;
   if(!GetIndicators(tmp1,tmp2,tmp3,atr,tmp5)) return;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(ask <= 0 || bid <= 0 || atr <= 0) return;

   double stopDist = atr * StopATRMultiplier;
   double takeDist = atr * TakeATRMultiplier;
   double lot = ComputeLotByRisk(stopDist);
   if(lot <= 0.0) return;

   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(20);

   bool ok = false;
   if(direction > 0)
   {
      double sl = ask - stopDist;
      double tp = ask + takeDist;
      ok = trade.Buy(lot, _Symbol, ask, sl, tp, "AI Buy");
      if(ok) lastEntryPrice = ask;
   }
   else
   {
      double sl = bid + stopDist;
      double tp = bid - takeDist;
      ok = trade.Sell(lot, _Symbol, bid, sl, tp, "AI Sell");
      if(ok) lastEntryPrice = bid;
   }

   if(ok)
   {
      ArrayCopy(lastX, x);
      hasLastSample = true;
      lastDirection = direction;
   }
}

int OnInit()
{
   hFastEMA = iMA(_Symbol, WorkTF, FastEMA, 0, MODE_EMA, PRICE_CLOSE);
   hSlowEMA = iMA(_Symbol, WorkTF, SlowEMA, 0, MODE_EMA, PRICE_CLOSE);
   hRSI     = iRSI(_Symbol, WorkTF, RSIPeriod, PRICE_CLOSE);
   hATR     = iATR(_Symbol, WorkTF, ATRPeriod);
   hADX     = iADX(_Symbol, WorkTF, ADXPeriod);

   if(hFastEMA == INVALID_HANDLE || hSlowEMA == INVALID_HANDLE || hRSI == INVALID_HANDLE || hATR == INVALID_HANDLE || hADX == INVALID_HANDLE)
      return(INIT_FAILED);

   RefreshDayState();
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   if(hFastEMA != INVALID_HANDLE) IndicatorRelease(hFastEMA);
   if(hSlowEMA != INVALID_HANDLE) IndicatorRelease(hSlowEMA);
   if(hRSI     != INVALID_HANDLE) IndicatorRelease(hRSI);
   if(hATR     != INVALID_HANDLE) IndicatorRelease(hATR);
   if(hADX     != INVALID_HANDLE) IndicatorRelease(hADX);
}

void OnTick()
{
   RefreshDayState();

   if(!IsNewBar()) return;
   if(!TradingHour()) return;
   if(DailyLossExceeded()) return;
   if(!SpreadOk()) return;

   if(CountOpenPositions() >= MaxOpenPositions)
   {
      TryTrainFromClosedSignal();
      return;
   }

   double x[];
   if(!BuildFeatures(x)) return;

   double pUp = PredictProb(x);

   if(pUp >= BuyThreshold)
      OpenTrade(1, x);
   else if(pUp <= SellThreshold)
      OpenTrade(-1, x);
}
