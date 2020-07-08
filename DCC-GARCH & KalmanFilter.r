library(rmgarch)
library(rugarch)



# GARCH(1,1) specification
garch11.spec = ugarchspec(mean.model = list(armaOrder = c(0,0)),variance.model = list(garchOrder = c(1,1), model = "sGARCH"),distribution.model = "norm")
garch11.spec



# DCC specification
dcc.garch11.spec = dccspec(uspec = multispec(replicate(2,garch11.spec)),dccOrder = c(1,1),distribution = "mvnorm")
dcc.garch11.spec



us = read.csv('US dataset.csv')
date_us = us$Date
sp500 = us$SP500
wti_us = us$WTI
covid_new_us = us$NEW_CASE
covid_total_us = us$CUM_CASE
sr = us$sr
cr = us$cr
vix = us$VIX


library(ggplot2)
library(dplyr)
library(patchwork) 
# 
data <- data.frame(day = as.Date(date_us), stock = sp500, commodity = wti_us)
p1 <- ggplot(data, aes(x=day, y=stock)) +
  geom_line(color = 'green', size = 0.2) +
  ggtitle('Stock Market') 
p2 <- ggplot(data, aes(x=day, y=commodity)) +
  geom_line(color = 'blue', size = 0.2) +
  ggtitle('Commodity Market') 
p1 + p2
```

```{r}
data <- data.frame(day = as.Date(date_us), stock = sr, commodity = cr)
p1 <- ggplot(data, aes(x=day, y=stock)) +
  geom_line(size = 0.1) +
  ggtitle('Stock Market Return') +
  xlab('time') +
  ylab('SP500')
p2 <- ggplot(data, aes(x=day, y=commodity)) +
  geom_line(size = 0.1) +
  ggtitle('Commodity Market Return') +
  xlab('time') +
  ylab('Crude Oil')
p1+p2



dcc.fit_us = dccfit(dcc.garch11.spec,data=data.frame(stock = sr, commodity = cr))
correlation_us = unname(rcor(dcc.fit_us, type = 'R')[1,2,])
plot(as.ts(correlation_us), ylab = 'Conditional Correlation')



cov_us = unname(rcov(dcc.fit_us)[1,2,])
vol_stock_us = unname(rcov(dcc.fit_us)[1,1,])
vol_commo_us = unname(rcov(dcc.fit_us)[2,2,])
new_us = covid_new_us
cum_us = covid_total_us
us_data = data.frame(date_us,correlation_us,cov_us,vol_stock_us,vol_commo_us,new_us,cum_us,vix)
write.csv(us_data,'correlation_us.csv')



plot(as.ts(vol_stock_us), ylab = 'Volatility SP500')
plot(as.ts(vol_commo_us), ylab = 'Volatility Crude Oil')



diff_new <- diff(covid_new_us[3279:3390])
plot(as.ts(diff_new))
adf.test(diff_new)
model.build=function(p){
  return(dlmModPoly(2,dV=p[1],dW=p[2:3]))
}
model.mle1=dlmMLE(diff_new,parm = c(0.1,0,1,1),
                 build = model.build)
#if(model.mle1$convergence==0) print("converged") else print("did not converge")
model.mle1$par
model.fit1 <- model.build(model.mle1$par)
model.filtered1 <- dlmFilter(diff_new, model.fit1)
n <- 5
model.forecast <- dlmForecast(model.filtered1, nAhead=n)
a <- drop(model.forecast$a%*%t(FF(model.fit1)))
df <- data.frame(x=c(1:5), y=a, series="forecast")
g.dlm <- ggplot(df, aes(x=x, y=y, colour=series)) + 
  geom_line()
g.dlm


forecast_case = c()
num = 0
cum = covid_new_us[3390]
for (i in a) {
  num = num + 1
  cum = floor(cum + i)
  forecast_case[num] = cum
}



plot(as.ts(vix))
adf.test(vix)
model.build=function(p){
  return(dlmModPoly(2,dV=p[1],dW=p[2:3]))
}
model.mle2=dlmMLE(vix,parm = c(0.1,0,1,1),
                 build = model.build)
model.mle2$par
model.fit2 <- model.build(model.mle2$par)
model.filtered2 <- dlmFilter(vix, model.fit2)
n <- 5
model.forecast2 <- dlmForecast(model.filtered2, nAhead=n)
a <- drop(model.forecast2$a%*%t(FF(model.fit2)))
df <- data.frame(x=c(1:5), y=a, series="forecast")
g.dlm <- ggplot(df, aes(x=x, y=y, colour=series)) + 
  geom_line()
g.dlm
forecast_vix = a




dcc.fcst_us = dccforecast(dcc.fit_us,n.ahead = 5)
cov_df = rcov(dcc.fcst_us)
vol_stock_us_fcst = unname(cov_df$`1979-04-13 19:00:00`[1,1,])
vol_commo_us_fcst = unname(cov_df$`1979-04-13 19:00:00`[2,2,])



us_forecast_case = data.frame(forecast_case,vol_stock_us_fcst,vol_commo_us_fcst,forecast_vix)
write.csv(us_forecast_case,'forecast_us.csv')
