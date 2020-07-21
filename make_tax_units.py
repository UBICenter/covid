""" Makes tax units from the ASEC.

Based on Sam Portnow's code at 
https://users.nber.org/~taxsim/to-taxsim/cps/cps-portnow/TaxSimRScriptForDan.R
"""
import numpy as np
import pandas as pd

# personal exemptions

pexemp <- pd.DataFrame({
  'year': np.arange(1960, 2014)
  'pexemp': [600,600,600,600,600,600,600,600,600,600,625,675,750,750,750,750,
             750,750,750, 1000,1000,1000,1000,1000,1000,1040,1080,1900,1950,
             2000,2050,2150,2300,2350,2450,2500,2550,2650,2700,2750,2800,2900,
             3000,3050,3100,3200,3300,3400,3500,3650,3750,3750,3750,3750]})

# gotta get 1999 thru 2009
pexemp = pexemp[pexemp.year.between(1998, 2010)]


ipum = pd.read_csv('~/MaxGhenis/datarepo/pppub19.csv.gz')
# set to lower case
ipum.columns = ipum.columns.str.lower()

# /* Set missing income items to zero so that non-filers etc will get zeroes.*/
# find out what statatax is and get it
VARS1 = ['eitcred', 'fedretir']
VARS2 = ['fedtax', 'statetax', 'adjginc', 'taxinc', 'fedtaxac', 'fica',
         'caploss', 'stataxac', 'incdivid', 'incint', 'incrent', 'incother',
         'incalim', 'incasist', 'incss', 'incwelfr', 'incwkcom', 'incvet',
         'incchild', 'incunemp', 'inceduc', 'gotveduc', 'gotvothe', 'gotvpens',
         'gotvsurv', 'incssi']
VARS3 <- ['incwage', 'incbus', 'incfarm', 'incsurv', 'incdisab', 'incretir']
vars <- VARS1 + VARS2 + VARS3 + ['capgain']


# these are the missing codes
MISSING_CODES = [9999, 99999, 999999, 9999999,
                 -9999, -99999, -999999, -9999999,
                 9997, 99997, 999997, 9999997]

for var in vars:
    ipum[var] = np.where(ipum[var].isna() or ipum[var].isin(MISSING_CODES), 0,
                         ipum[var])

# set 0's to NA for location
COLS_ZERO_TO_NA = ['momloc', 'poploc', 'sploc']
for col in COLS_ZERO_TO_NA:
    ipum[col] = np.where(ipum[col] == 0, np.nan, ipum[col])


# year before tax returns
ipum['x2'] = ipum.year - 1

# set x3 to  fips code
ipum['x3'] <- ipum.statefip

# convert to soi - TODO
# source('FIPStoSOI.R')

# /* Marital status will be sum of spouse's x4 values */
ipum['x4' = 1

# if age > 65, x6 is 1
ipum['x6'] = np.where(ipum.age > 65, 1, 0)

# primary wage or spouse wage
ipum['incwagebusfarm'] = ipum[['incwage', 'incbus', 'incfarm']].sum(axis=1)
ipum['x7'] = np.where(ipum.sex == 1, ipum.incwagebusfarm, 0)
ipum['x8'] = np.where(ipum.sex == 2, ipum.incwagebusfarm, 0)


ipum['x9'] = ipum.incdivid
ipum['x10'] = ipum[['incint', 'incrent', 'incother', 'incalim']].sum(axis=1)
ipum['x11'] = ipum.incretir
ipum['x12'] = ipum.incss

# /* Commented out got* items below because they are an error - 
# hope to fix soon. drf, 
# Nov18, 2015
# */
ipum['x13'] <- ipum[['incwelfr', 'incwkcom', 'incvet', 'incsurv', 'incdisab',
                     'incchild', 'inceduc', 'incssi', 'incasist']].sum(axis=1)
  # /*gotveduc+gotvothe+gotvpens+gotvsurv+*/ 

ipum['x14'] = 0
ipum['x15'] = 0


# /* use Census imputation of itemized deductions where available.*/
# first have to join the exemption table
pexemp[1] <- 'x2'
ipum <- join(ipum, pexemp, by='x2')

# adjusted gross - taxes + exemptions
ipum$x16 <- ipum$adjginc - rowSums(ipum[,c('pexemp', 'proptax', 'statetax', 'taxinc')], na.rm=T)
# no values less than 0
ipum$x16 <- ifelse(ipum$x16 < 0, 0, ipum$x16)

ipum$x17 <- 0
ipum$x18 <- ipum$incunemp
ipum$x19 <- 0
ipum$x20 <- 0
ipum$x21 <- 0

# * Assume capgain and caploss are long term;
ipum$x22 <- ifelse(! ipum$capgain==-999, ipum$capgain - ipum$caploss, 0)
ipum$capgain[ipum$capgain==-999] <- 0


# Here we output a record for each person, so that tax units can be formed 
# later by summing over person records. The taxunit id is the minimum of
# the pernum or sploc, so spouses will get the same id. For children
# it is the minimum of the momloc or poploc. Other relatives are made
# dependent on the household head (which may be incorrect) and non-relatives
# are separate tax units. 
# */
attach(ipum)

ipum$hnum <- 0
ipum[ipum$relate==101,]$hnum <- ipum[ipum$relate==101,]$pernum
ipum[ipum$hnum==0,]$hnum <- NA

# if claiming > personal exemption than they're their own filer
ipum$notself <- ifelse((ipum$x7 + ipum$x8 + ipum$x9 + ipum$x10 + ipum$x11 + ipum$x12 + ipum$x13 + ipum$x22) <= ipum$pexemp, 1, 0)

ipum$sum <- rowSums(ipum[,c('x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x22')], na.rm=T)

ipum$notself <- ifelse(rowSums(ipum[,c('x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x22')], na.rm=T) <= ipum$pexemp, 1, 0)

ipum$deprel <- ifelse(ipum$notself==1 & ipum$relate >= 303 & ipum$relate <= 901, 1, 0)
ipum$depchild <- ifelse(ipum$notself==1 & rowSums(ipum[,c('momloc', 'poploc')], na.rm=T) > 0 & (ipum$age < 18 | ipum$age < 24 & ipum$schlcoll > 0), 1, 0)

detach(ipum)

# set dependents and taxpayers
dpndnts <- ipum[ipum$depchild==1 | ipum$deprel==1,]
dpndnts$x1 <- 0
dpndnts[dpndnts$depchild == 1,]$x1 <- 100*dpndnts[dpndnts$depchild == 1,]$serial + apply(dpndnts[dpndnts$depchild == 1,][,c('momloc', 'poploc')], 1, min, na.rm=T)
dpndnts[dpndnts$deprel == 1,]$x1 <- 100*dpndnts[dpndnts$deprel == 1,]$serial + dpndnts[dpndnts$deprel==1,]$hnum

dpndnts$x4 <- NA
dpndnts$x5 <- 1
dpndnts$x6 <- NA
dpndnts$x19 <- NA

txpyrs <- ipum[ipum$depchild == 0 & ipum$deprel == 0,]
txpyrs$x1 <- 0
txpyrs$x1 <- 100*txpyrs$serial + apply(txpyrs[,c('pernum', 'sploc')], 1, min, na.rm=T)

txpyrs$x5 <- 0


# set whats not x1, x2, or x5 in deps to NA
vars <- c('x3', paste0('x', 4), paste0('x', 6:22))
dpndnts[,vars] <- NA

# put them back together
ipum <- rbind(txpyrs, dpndnts)


library(dplyr)
# sum value over tax #
concat <- group_by(ipum, x2, x1)

concat <- summarise(concat, x3 = mean(x3, na.rm=T), x4 = sum(deprel, na.rm=T) + sum(depchild, na.rm=T),
                    x5 = sum(x5, na.rm=T), x6 = sum(x6, na.rm=T), x7 = sum(x7, na.rm=T), x8 = sum(x8, na.rm=T), x9 = sum(x9, na.rm=T),
                    x10 = sum(x10, na.rm=T), x11 = sum(x11, na.rm=T), x12 = sum(x12, na.rm=T), x13 = sum(x13, na.rm=T), x14 = sum(x14, na.rm=T), x15 = sum(x15, na.rm=T),
                    x16 = sum(x16, na.rm=T), x17 = sum(x17, na.rm=T), x18 = sum(x18, na.rm=T), x19 = sum(depchild, na.rm=T), x20 = sum(x20, na.rm=T), x21 = sum(x21, na.rm=T), x22 = sum(x22, na.rm=T))

concat <- data.frame(concat)

concat[! is.na(concat$x4) & concat$x4 == 1 & concat$x5 > 0,]$x4 <- 3

concat <- concat[concat$x19 > 0,]
concat <- concat[concat$x4 > 0, ]
vars <- paste0('x', 1:22)

concat <- concat[,vars]

names(concat) <- c('id', 'year', 'state', 'mstat', 'depx', 'agex', 'pwages', 'swages', 'dividends', 'otherprop', 'pensions', 'gssi', 'transfers', 'rentpaid', 'proptax', 
                   'otheritem', 'childcare', 'ui', 'depchild', 'mortgage', 'stcg', 'ltcg')