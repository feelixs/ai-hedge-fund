SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'AMD', 'INTC', 'SPY', 'QQQ']

# Medium-sized list of symbols (50)
MED_SYMBOLS = ['AAPL', 'ABNB', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AMAT', 'AMD', 'AMGN', 'AMZN', 'ASML',
               'AVGO', 'BKNG', 'CDNS', 'CMCSA', 'COST', 'CSCO', 'GOOG', 'GOOGL', 'INTC', 'F',
               'INTU', 'ISRG', 'LRCX', 'MELI', 'META', 'MSFT', 'NFLX', 'NVDA', 'PYPL', 'QCOM', 'LCID',
               'DUK', 'MMM', 'SBUX', 'TXN', 'TSLA', 'SPY', 'QQQ', 'AIG', 'NIO', 'PLTR', 'QBTS', 'IQ',
               'SNAP', 'SMCI', 'NU', 'TSM', 'BABA', 'BMY', 'LPSN']

LARGE_SYMBOLS = ['AAPL', 'ABNB', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AMAT', 'AMD', 'AMGN', 'AMZN', 'ASML', 'EXEL', 'LCTX',
                 'AVGO', 'BKNG', 'CDNS', 'CMCSA', 'COST', 'CSCO', 'GOOG', 'GOOGL', 'INTC', 'F', 'INTU', 'INO', 'RAPT',
                 'ISRG', 'LRCX', 'MELI', 'META', 'MSFT', 'NFLX', 'NVDA', 'PYPL', 'QCOM', 'LCID', 'DUK', 'TSHA', 'BIGC',
                 'MMM', 'SBUX', 'TXN', 'TSLA', 'SPY', 'QQQ', 'AIG', 'NIO', 'PLTR', 'QBTS', 'IQ', 'SNAP', 'GH', 'BAND',
                 'SMCI', 'NU', 'TSM', 'BABA', 'BMY', 'LPSN', 'TMUS', 'CRWD', 'ZM', 'ROKU', 'MTCH', 'OKTA', 'GM', 'CRBU',
                 'ZS', 'DDOG', 'NET', 'MDB', 'DOCU', 'PDD', 'JD', 'BILI', 'BIDU', 'NTES', 'WDAY', 'TWLO', 'GME', 'TSVT',
                 'TEAM', 'CRM', 'NOW', 'TTD', 'PINS', 'ETSY', 'SPOT', 'DASH', 'RIVN', 'RBLX', 'U', 'COIN', 'MP', 'NTRA',
                 'SNOW', 'SHOP', 'UBER', 'LYFT', 'PTON', 'CHWY', 'ZI', 'DT', 'DKNG', 'CRSR', 'HOOD', 'SOFI', 'LI', 'BHC',
                 'AFRM', 'UPST', 'FUBO', 'ROOT', 'CLOV', 'ME', 'OPEN', 'OSCR', 'PATH', 'IONQ', 'DNA', 'ENVX', 'AI', 'GO',
                 'LAZR', 'CHPT', 'BLNK', 'FCEL', 'PLUG', 'BE', 'ENPH', 'SEDG', 'RUN', 'NOVA', 'LAC', 'SHLS', 'FSLR',
                 'XPEV', 'FFIE', 'GOEV', 'NKLA', 'WKHS', 'HYLN', 'DM', 'MKFG', 'XMTR', 'CFLT', 'GLBE', 'GTLB', 'ASAN',
                 'MNDY', 'CRDO', 'WRBY', 'DOCS', 'BZFD', 'SMRT', 'BIRD', 'OUST', 'JOBY', 'LILM', 'ACHR', 'EVGO', 'QS',
                 'STEM', 'GEVO', 'CLNE', 'MARA', 'DCFC', 'SLDP', 'RIOT', 'CIFR', 'HUT', 'BTBT', 'BITF', 'BITO', 'GBTC',
                 'ETHE', 'SGMO', 'CRSP', 'EDIT', 'NTLA', 'BEAM', 'VERV', 'VCYT', 'EXAS', 'PACB', 'BNGO', 'RXRX', 'FATE',
                 'TWST', 'ABCL', 'GDRX', 'CERT', 'ACVA', 'COUR', 'DUOL', 'HIMS', 'ELAN', 'ALEC', 'SANA', 'SRNE', 'NVAX',
                 'MRNA', 'BNTX', 'VXRT', 'OCGN', 'GERN', 'CMRX', 'SAVA', 'CRNC', 'CRNX', 'ARGX', 'KRYS', 'NARI', 'TVTX',
                 'AVXL', 'AXSM', 'IONS', 'JAZZ', 'SAGE', 'ACAD', 'VTRS', 'ZTS', 'VRTX', 'REGN', 'ALNY', 'ILMN', 'BIIB',
                 'GILD', 'IDXX', 'DXCM', 'ALGN', 'VEEV', 'HOLX', 'BMRN', 'UTHR', 'RGEN', 'NBIX', 'TECH', 'ICLR', 'XRAY',
                 'TGTX', 'FOLD', 'DVAX', 'RARE', 'CDXS', 'ITCI', 'RCUS', 'STRO', 'MGNX', 'KURA', 'KYMR', 'ATRA', 'LYEL',
                 'CNTA', 'OMGA', 'NOTV', 'PRCT', 'RXST', 'ITOS', 'BDTX', 'LUNG', 'PROC', 'ALLK', 'IMVT', 'ABUS', 'AFMD',
                 'CARA', 'ANNX', 'CCCC', 'PRTA', 'GLPG', 'GMAB', 'BPMC', 'ATRC', 'NSTG', 'QDEL', 'NVTS', 'RPAY', 'SEAC',
                 'CARG', 'EGHT', 'ALTR', 'QLYS', 'SAIL', 'APPS', 'BLKB', 'CCOI', 'CNXN', 'CSGS', 'CYBR', 'DBX', 'DOMO',
                 'DTSS', 'EBAY', 'EGRX', 'ELF', 'ELTK', 'ENLV', 'ENPH', 'ENTG', 'EPAM', 'EQIX', 'EQBK', 'ERIC', 'ESLT',
                 'ESTC', 'ETSY', 'EVRI', 'EXEL', 'EXLS', 'EXPE', 'EXPO', 'EXTR', 'FANG', 'FBIO', 'FBIZ', 'FBNC', 'FCBC',
                 'FCEL', 'FCFS', 'FCNCA', 'FCPT', 'FDBC', 'FDUS', 'FEIM', 'FELE', 'FFBC', 'FFIC', 'FFIN', 'FFIV', 'FFNW',
                 'FFWM', 'FGBI', 'FGEN', 'FIBK', 'FITB', 'FIVE', 'FIVN', 'FIZZ', 'FLGT', 'FLIC', 'FLNT', 'FLWS', 'FMBH',
                 'FMNB', 'FNKO', 'FNLC', 'FHB', 'FOLD', 'FORM', 'FORR', 'FOSL', 'FOX', 'FOXA', 'FOXF', 'FRBA', 'FRHC',
                 'FRME', 'FROG', 'FRPH', 'FRPT', 'FRSX', 'FSBW', 'FSEA', 'FSFG', 'FSLR', 'FTDR', 'FTEK', 'FTFT', 'FTGC',
                 'FTHI', 'FTNT', 'FTRI', 'FULT', 'FUNC', 'FUSB', 'FUTU', 'FWONA', 'FWONK', 'FWRD', 'FXNC', 'GABC', 'GEG',
                 'GAIN', 'GAINL', 'GALT', 'GAME', 'GAN', 'GANX', 'GASS', 'GBCI', 'GBDC', 'GBIO', 'GCBC', 'GDEN', 'GDRX',
                 'GDS', 'GDYN', 'GECC', 'GAIA', 'GENC', 'GENE', 'GEOS', 'GERN', 'GEVO', 'GGAL',  'GHSI', 'GIFI', 'GIII',
                 'GILD', 'GILT', 'GLBS', 'GLBZ', 'GLDD', 'GLNG', 'GLPG', 'GLPI', 'GLRE', 'GLYC', 'GNFT', 'GNLN', 'GNMA',
                 'GNOM', 'GNTX', 'GNTY', 'GOCO', 'GOGL', 'GOGO', 'GOOD', 'GOODO', 'GOOG', 'GOOGL', 'GOSS', 'GOVX', 'GP',
                 'GPMT', 'GPRE', 'GPRO', 'GRBK', 'GRFS', 'GRPN', 'GRTX', 'GRVY', 'GRWG', 'GSBC', 'GSHD', 'GSIT', 'GSM',
                 'GTBP', 'GTEC', 'GTIM', 'GTLS', 'GTLS', 'GTLS']


tickers = LARGE_SYMBOLS
