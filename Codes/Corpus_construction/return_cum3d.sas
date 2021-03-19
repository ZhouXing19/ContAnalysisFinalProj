/* This file compute three day returns around 10-K announcement date */

data crsp_d;
set crspm.dsf;
where date >= "01jan2010"d;
run;

data crsp_d;
set crsp_d;
keep cusip permno date ret;
run;

data ff;
set ff.factors_daily;
run;

data ff;
set ff;
where date >= "01jan2010"d;
keep date mktrf;
run;

proc sql;
create table crsp_d_mkt
as select a.*, b.* 
from crsp_d a, ff b
where a.date = b.date;
quit;

data crsp_d_mkt;
set crsp_d_mkt;
rename mktrf = vwret;
run;

proc sort data = crsp_d_mkt;
by permno date;
run;

data t_1;
set crsp_d_mkt;
by permno;
date = lag(date);
if first.permno then delete;
rename ret = ret_1;
rename vwret = vwret_1;
run;

/* data t_1 / view=t_1; */
/* set crsp_d_mkt; */
/* date = intnx("day", date, +1); */
/* rename ret=ret_1 vwret=vwret_1; */
/* run; */

data t1;
set crsp_d_mkt;
rename ret = ret1;
rename vwret = vwret1;
run;

proc sort data = t1;
by permno descending date;
run;

data t1;
set t1;
by permno;
date = lag(date);
if first.permno then delete;
run;

proc sort data = t1;
by permno date;
run;

/* data t1 / view=t1; */
/* set crsp_d_mkt; */
/* date = intnx("day", date, -1); */
/* rename ret=ret1 vwret=vwret1; */
/* run; */

data want;
merge t_1 crsp_d_mkt(in=ok) t1;
by permno date;
if ok;
if n(of ret:) = 3 then
    ma_ret = (ret_1+1)*(ret+1)*(ret1+1) -
        (vwret_1+1)*(vwret+1)*(vwret1+1);
keep permno date ret vwret ma_ret;
run;

/* Get the link from gvkey to permno */

proc sql;
create table ccm
as select gvkey, lpermno as permno, linktype, linkprim, linkdt, linkenddt
from crsp.ccmxpf_linktable
where substr(linktype,1,1)='L'
and (linkprim ='C' or linkprim='P');
quit;

proc import datafile = '/home/uchicago/songrunhe/gvkey_date.csv'
out = gvkey_date
dbms = csv
replace;
run;

data gvkey_date;
set gvkey_date;
FDATE = input( Put( FDATE, 8.), YYMMDD10.); 
format FDATE YYMMDDN8.;
run;

data ccm;
set ccm;
gvkey_num = input(gvkey, 6.0);
run;

proc sql;
create table gvkey_date_permno
as select a.*, b.permno
from gvkey_date a, ccm b
where a.gvkey = b.gvkey_num and (intnx('month', intnx('year', a.fdate, 0, 
		'E'), 6, 'E') >=b.linkdt or missing(b.linkdt)) 
		and (b.linkenddt >=intnx('month', intnx('year', a.fdate, 0, 'E'), 6, 'E') 
		or missing(b.linkenddt));
quit;

proc sort data = gvkey_date_permno;
by gvkey fdate permno;
run;

proc sql;
create table gvkey_date_permno_ret3d_ret
as select a.*, b.ma_ret, b.ret, b.vwret
from gvkey_date_permno a, want b
where a.permno = b.permno and a.fdate = b.date;
quit;

proc export data = gvkey_date_permno_ret3d_ret
dbms = csv
outfile = '/home/uchicago/songrunhe/gvkey_date_ret.csv'
replace;
run;


