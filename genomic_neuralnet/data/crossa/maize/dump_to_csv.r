setwd('~/projects/genomic_sel/crossa/analysis_wheat_corn/corn_flowering/data')

# Get all of the datasets in the directory.
datasets <- c( 'dataCorn_SS_asi.RData'
             , 'dataCorn_SS_flf.RData'
             , 'dataCorn_SS_flm.RData'
             , 'dataCorn_WW_asi.RData'
             , 'dataCorn_WW_flf.RData'
             , 'dataCorn_WW_flm.RData'
             )

# Grab the raw alleles.
load(datasets[1])
raw_data = X

# Pull out the value of 'y' from each dataset, which is the phenotype.
load(datasets[1])
ss_asi = y
load(datasets[2])
ss_flf = y
load(datasets[3])
ss_flm = y
load(datasets[4])
ww_asi = y
load(datasets[5])
ww_flf = y
load(datasets[6])
ww_flm = y

# Make a phenotypes csv file.
df <- data.frame(ss_asi=ss_asi, ss_flf=ss_flf, ss_flm=ss_flm, ww_asi=ww_asi, ww_flf=ww_flf, ww_flm=ww_flm)
write.csv(df, file='phenotypes.csv')

# Make a genotypes csv file.
write.table(t(X), file='genotypes.csv', row.names=FALSE, col.names=FALSE, sep=',')
