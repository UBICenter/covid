"""
Must run convert_asec_taxcalc.py and make_tax_units.py first.
"""

# Set columns to lowercase and to 0 or null as appropriate.
prep_ipum(person)
# Add taxid and related fields.
tax_unit_id(person)
# Add other person-level columns in taxcalc form.
person = convert_asec_person_taxcalc(person)
# 99 is the missing code for wksunem1.
# Note: Missing codes for features used in taxcalc are recoded in
# convert_asec_taxcalc.py.
person.loc[person.wksunem1 == 99, 'wksunem1'] = 0
# The 2014 file was released in two ways, so weights must be halved.
person.asecwt *= np.where(person.year == 2014, 0.5, 1)
person.spmwt *= np.where(person.year == 2014, 0.5, 1)

# Adjust all dollar values to 2020.
cpiu = pdr.get_data_fred('CPIAUCSL', '2009-01-01')
# Filter to January.
cpiu = cpiu[cpiu.index.month == 1]
cpiu.index = cpiu.index.year
# Multiplier to 2020.
cpiu['inflate2020'] = cpiu.CPIAUCSL[2020] / cpiu.CPIAUCSL
# Survey year is the reported year minus 1.
person['FLPDYR'] = person.year - 1
person = person.merge(cpiu[['inflate2020']], left_on='FLPDYR',
                      right_index=True)
DOLLAR_VALS = ['spmtotres', 'spmthresh',
               'e02400', 'e01700', 'e00300', 'e02300', 'e00650', 'incrent',
               'p23250', 'fica', '_e00200', 'e00600', 'e01500', 
               'e00200p', 'e00200s', 'e00200']
for i in DOLLAR_VALS:
    person[i] = person[i] * person.inflate2020

"""
## Add UI to person records

Assume that unemployment blocks are contiguous and randomly distributed.
"""
person['ui_start'] = np.random.randint(1, 53 - person.wksunem1,
                                       person.shape[0])
person['ui_end'] = person.ui_start + person.wksunem1

FPUC_START = 13  # April was the 13th week.
FPUC_MAX_WEEKS = 17  # April to July.
FPUC2_START = FPUC_START + FPUC_MAX_WEEKS
FPUC2_MAX_WEEKS = 22  # August to December.
FPUC_WEEKLY_BEN = 600
person['fpuc_weeks'] = np.fmax(
    0, np.fmin(person.ui_end - FPUC_START,
               np.fmin(person.wksunem1, FPUC_MAX_WEEKS)))
person['fpuc2_weeks'] = np.fmax(
    0, np.fmin(person.ui_end - FPUC2_START,
               np.fmin(person.wksunem1, FPUC2_MAX_WEEKS)))
person['fpuc'] = FPUC_WEEKLY_BEN * person.fpuc_weeks
person['fpuc2'] = person.fpuc + FPUC_WEEKLY_BEN * person.fpuc2_weeks

# Checks
assert person.fpuc_weeks.max() == FPUC_MAX_WEEKS
assert person.fpuc2_weeks.max() == FPUC2_MAX_WEEKS
assert person.fpuc_weeks.min() == person.fpuc2_weeks.min() == 0

# Store original unemployment benefits.
person['e02300_orig'] = person.e02300

"""
## Create tax units and calculate tax liability
"""
person['RECID'] = person.FLPDYR * 1e9 + person.taxid

def get_taxes(tu):
    """ Calculates taxes by running taxcalc on a tax unit DataFrame.
    
    Args:
        tu: Tax unit DataFrame.
    
    Returns:
        Series with tax liability for each tax unit.
    """
    return mdf.calc_df(records=tc.Records(tu, weights=None, gfactors=None),
                       # year doesn't matter without weights or gfactors.
                       year=2020).tax.values

# Create tax unit dataframe.
tu = create_tax_unit(person)
tu['tax'] = get_taxes(tu)

# Simulate FPUC.

# Create tax unit dataframe.
person.e02300 = person.e02300_orig + person.fpuc
tu_fpuc = create_tax_unit(person)
tu['e02300_fpuc'] = tu_fpuc.e02300
tu['tax_fpuc'] = get_taxes(tu_fpuc)
del tu_fpuc

# Simulate extended FPUC.

# Create tax unit dataframe.
person.e02300 = person.e02300_orig + person.fpuc2
tu_fpuc2 = create_tax_unit(person)
tu['e02300_fpuc2'] = tu_fpuc2.e02300
tu['tax_fpuc2'] = get_taxes(tu_fpuc2)
del tu_fpuc2

# Change person e02300 back.
person.e02300 = person.e02300_orig

"""
## Merge back to the person level

Have each person pay the share of tax differences in proportion with their
FPUC.
"""

tu['fpuc_total'] = tu.e02300_fpuc - tu.e02300
tu['fpuc2_total'] = tu.e02300_fpuc2 - tu.e02300
tu['fpuc_tax_total'] = tu.tax_fpuc - tu.tax
tu['fpuc2_tax_total'] = tu.tax_fpuc2 - tu.tax

person = person.merge(tu[['RECID', 'fpuc_total', 'fpuc2_total',
                          'fpuc_tax_total', 'fpuc2_tax_total']],
                      on='RECID')

for i in ['fpuc', 'fpuc2']:
    person[i + '_tax'] = np.where(person[i + '_total'] == 0, 0,
        person[i + '_tax_total'] * person[i] / person[i + '_total'])
    person[i + '_net'] = person[i] - person[i + '_tax']
    
# Checks that the totals match by person and tax unit, then garbage-collect.
assert np.allclose(tu.fpuc_total.sum(), person.fpuc.sum())
assert np.allclose(tu.fpuc2_total.sum(), person.fpuc2.sum())
assert np.allclose(tu.fpuc_tax_total.sum(), person.fpuc_tax.sum())
assert np.allclose(tu.fpuc2_tax_total.sum(), person.fpuc2_tax.sum())
del tu

"""
## Calculate budget-neutral UBIs and payroll taxes
"""

def single_year_summary(year):
    fpuc_budget = mdf.weighted_sum(person[person.FLPDYR == year],
                                   'fpuc_net', 'asecwt')
    fpuc1_2_budget = mdf.weighted_sum(person[person.FLPDYR == year],
                                      'fpuc2_net', 'asecwt')
    fpuc2_budget = fpuc1_2_budget - fpuc_budget
    pop = person[person.FLPDYR == year].asecwt.sum()
    adult_pop = person[(person.FLPDYR == year) &
                       (person.age > 17)].asecwt.sum()
    total_fica = mdf.weighted_sum(person[person.FLPDYR == year],
                                  'fica', 'asecwt')
    fpuc_ubi = fpuc_budget / pop
    fpuc_adult_ubi = fpuc_budget / adult_pop
    fpuc_fica_pct_cut = 100 * fpuc_budget / total_fica
    # Note: FPUC2 includes FPUC1.
    fpuc2_ubi = fpuc2_budget / pop
    fpuc2_adult_ubi = fpuc2_budget / adult_pop
    fpuc2_fica_pct_cut = 100 * fpuc2_budget / total_fica
    return pd.Series([fpuc_budget, fpuc2_budget, pop, adult_pop, total_fica,
                      fpuc_ubi, fpuc_adult_ubi, fpuc_fica_pct_cut,
                      fpuc2_ubi, fpuc2_adult_ubi, fpuc2_fica_pct_cut])

OVERALL_YEARLY_METRICS = ['fpuc_budget', 'fpuc2_budget', 'pop', 'adult_pop',
                          'total_fica']
FPUC_YEARLY_METRICS = ['fpuc_ubi', 'fpuc_adult_ubi', 'fpuc_fica_pct_cut']
FPUC2_YEARLY_METRICS = ['fpuc2_ubi', 'fpuc2_adult_ubi', 'fpuc2_fica_pct_cut']
all_metrics = (
    OVERALL_YEARLY_METRICS + FPUC_YEARLY_METRICS + FPUC2_YEARLY_METRICS)
DISPLAY_METRICS = {
    'fpuc_budget': 'Cost of FPUC',
    'fpuc2_budget': 'Cost of expanding FPUC',
    'pop': 'Population',
    'adult_pop': 'Adult population',
    'total_fica': 'Total FICA',
    'fpuc_ubi': 'Universal one-time payment (FPUC)',
    'fpuc_adult_ubi': 'Adult one-time payment (FPUC)',
    'fpuc_fica_pct_cut': 'FICA % cut (FPUC)',
    'fpuc2_ubi': 'Universal one-time payment (FPUC2)',
    'fpuc2_adult_ubi': 'Adult one-time payment (FPUC2)',
    'fpuc2_fica_pct_cut': 'FICA % cut (FPUC2)'
}

year_summary = pd.DataFrame({'FLPDYR': person.FLPDYR.unique()})
year_summary[all_metrics] = year_summary.FLPDYR.apply(single_year_summary)

person = person.merge(
    year_summary[['FLPDYR'] + FPUC_YEARLY_METRICS + FPUC2_YEARLY_METRICS],
    on='FLPDYR')

"""
Run calculations on all fields (except `fpuc_ubi` which already works).
"""

# Zero out adult UBIs for children.
person.loc[person.age < 18, 'fpuc_adult_ubi'] = 0
# Calculate total FICA cut by multiplying FICA by % cut.
# Divide by 100 as it was previously multiplied by 100 for table displaying.
person['fpuc_fica_cut'] = person.fica * person.fpuc_fica_pct_cut / 100
# Similar process for FPUC2, but also adding fpuc_net since this is on top
# of the existing FPUC.
person['fpuc2_ubi'] = person.fpuc_net + person.fpuc2_ubi
person['fpuc2_adult_ubi'] = (person.fpuc_net + 
                             np.where(person.age > 17,
                                      person.fpuc2_adult_ubi, 0))
person['fpuc2_fica_cut'] = (person.fpuc_net +
                             person.fica * person.fpuc2_fica_pct_cut / 100)

"""
Verify the `fpuc` and `fpuc2` have equal costs, respectively, in each year.
"""
for year in person.FLPDYR.unique():
    tmp = person[person.FLPDYR == year]
    fpuc = mdf.weighted_sum(tmp, 'fpuc_net', 'asecwt')
    assert np.allclose(fpuc, mdf.weighted_sum(tmp, 'fpuc_ubi', 'asecwt'))
    assert np.allclose(fpuc, 
                       mdf.weighted_sum(tmp, 'fpuc_adult_ubi', 'asecwt'))
    assert np.allclose(fpuc, mdf.weighted_sum(tmp, 'fpuc_fica_cut', 'asecwt'))
    fpuc2 = mdf.weighted_sum(tmp, 'fpuc2_net', 'asecwt')
    assert np.allclose(fpuc2, mdf.weighted_sum(tmp, 'fpuc2_ubi', 'asecwt'))
    assert np.allclose(fpuc2, 
                       mdf.weighted_sum(tmp, 'fpuc2_adult_ubi', 'asecwt'))
    assert np.allclose(fpuc2, mdf.weighted_sum(tmp,
                                               'fpuc2_fica_cut', 'asecwt'))
del tmp

"""
## Aggregate to SPM units
"""

SPM_COLS = ['FLPDYR', 'spmfamunit', 'spmtotres', 'spmthresh', 'spmwt']
CHG_COLS = ['fpuc_net', 'fpuc_ubi', 'fpuc_adult_ubi', 'fpuc_fica_cut',
            'fpuc2_net', 'fpuc2_ubi', 'fpuc2_adult_ubi', 'fpuc2_fica_cut']
spmu = person.groupby(SPM_COLS)[CHG_COLS].sum().reset_index()
for i in CHG_COLS:
    spmu['spmtotres_' + i] = spmu.spmtotres + spmu[i]
    
"""
## Map back to persons
"""
spm_resource_cols = ['spmtotres_' + i for i in CHG_COLS]
SPMU_MERGE_COLS = ['spmfamunit', 'FLPDYR']
person = person.merge(spmu[SPMU_MERGE_COLS + spm_resource_cols],
                      on=SPMU_MERGE_COLS)
# Poverty flags.
for i in CHG_COLS:
    person['spmpoor_' + i ] = person['spmtotres_' + i] < person.spmthresh
# Also calculate baseline.
person['spmpoor'] = person.spmtotres < person.spmthresh

SPM_OUTCOLS = SPM_COLS + spm_resource_cols
spmu = spmu[SPM_OUTCOLS]

PERSON_OUTCOLS = (['asecwt', 'age', 'race', 'sex', 'diffany', 'spmpoor'] + 
                  CHG_COLS + spm_resource_cols + SPM_COLS +
                  ['spmpoor_' + i for i in CHG_COLS])
person = person[PERSON_OUTCOLS]


def pov(reform, year, age_group='All', race='All', disability_filter=False):
    """ Calculate the poverty rate under the specified reform for the
        specified population.
        Note: All arguments refer to the poverty population, not the reform.
    
    Args:
        reform: One of CHG_COLS. If None, provides the baseline rate.
        year: Year of the data (year before CPS survey year).
        age_group: Age group, either
            - 'Children' (under 18)
            - 'Adults' (18 or over)
            - 'All'
        race: Race code to filter to. Defaults to 'All'.
        disability_filter: Whether to filter for people with disabilities,
            i.e. DIFFANY == 2. The DIFFANY CPS IPUMS field
            is available for civilians aged 15+, and is 0 for NIU, 1 for
            no difficulty reported, 2 for some difficulty reported.
            Defaults to False.
        
    Returns:
        2018 SPM poverty rate.
    """
    # Select the relevant poverty column for the reform.
    if reform == 'baseline':
        poverty_col = 'spmpoor'
    else:
        poverty_col = 'spmpoor_' + reform
    # Filter by year.
    target_persons = person[person.FLPDYR == year]
    # Filter by age group.
    if age_group == 'Children':
        target_persons = target_persons[target_persons.age < 18]
    elif age_group == 'Adults':
        target_persons = target_persons[target_persons.age >= 18]
    # Filter by race.
    if race != 'All':
        target_persons = target_persons[target_persons.race == race]
    if disability_filter:
        target_persons = target_persons[target_persons.diffany == 2]
    # Return poverty rate (weighted average of poverty flag).
    return mdf.weighted_mean(target_persons, poverty_col, 'asecwt')


def pov_row(row):
    """ Calculate poverty based on parameters specified in the row.
    
    Args:
        row: pandas Series.
        
    Returns:
        2018 SPM poverty rate.
    """
    return pov(row.reform, row.year, row.age_group, row.race)

pov_rates = mdf.cartesian_product({'reform': ['baseline'] + CHG_COLS,
                                   'year': person.FLPDYR.unique(),
                                   'age_group': ['All', 'Children', 'Adults'],
                                   'race': ['All', 200],  # 200 means Black.
                                   'disability_filter': [True, False]
                                  })  
pov_rates['pov'] = 100 * pov_rates.apply(pov_row, axis=1)

"""
### Poverty gap and inequality
Calculate these for all people and SPM units, without breaking out by age or
race.
"""

def pov_gap_b(reform, year):
    if reform == 'baseline':
        resource_col = 'spmtotres'
    else:
        resource_col = 'spmtotres_' + reform
    tmp = spmu[spmu.FLPDYR == year]
    pov_gap = np.maximum(tmp.spmthresh - tmp[resource_col], 0)
    return (pov_gap * tmp.spmwt).sum() / 1e9

def pov_gap_row(row):
    return pov_gap_b(row.reform, row.year)

pov_gap_ineq = pov_rates[['reform', 'year']].drop_duplicates()
pov_gap_ineq['pov_gap_b'] = pov_gap_ineq.apply(pov_gap_row, axis=1)

"""
### Inequality

By individual based on their percentage of SPM resources.
"""

def gini(reform, year):
    if reform == 'baseline':
        resource_col = 'spmtotres'
    else:
        resource_col = 'spmtotres_' + reform
    tmp = person[person.FLPDYR == year]
    return mdf.gini(tmp[resource_col], tmp.asecwt)

def gini_row(row):
    return gini(row.reform, row.year)

pov_gap_ineq['gini'] = pov_gap_ineq.apply(gini_row, axis=1)

"""
## Postprocess

Create columns for displaying and grouping each reform.
"""

REFORM_DISPLAY = {
    'baseline': 'Baseline',
    'fpuc_net': '$600 per week UI',
    'fpuc_ubi': 'Payment to everyone',
    'fpuc_adult_ubi': 'Payment to adults',
    'fpuc_fica_cut': 'Payroll tax cut',
    'fpuc2_net': 'Extend $600 per week',
    'fpuc2_ubi': 'Payment to everyone',
    'fpuc2_adult_ubi': 'Payment to adults',
    'fpuc2_fica_cut': 'Payroll tax cut'
}

REFORM_GROUP = {
    'baseline': 'Baseline',
    'fpuc_net': 'fpuc',
    'fpuc_ubi': 'fpuc',
    'fpuc_adult_ubi': 'fpuc',
    'fpuc_fica_cut': 'fpuc',
    'fpuc2_net': 'fpuc2',
    'fpuc2_ubi': 'fpuc2',
    'fpuc2_adult_ubi': 'fpuc2',
    'fpuc2_fica_cut': 'fpuc2'
}

for i in [pov_rates, pov_gap_ineq]:
    i['reform_display'] = i.reform.map(REFORM_DISPLAY)
    i['reform_group'] = i.reform.map(REFORM_GROUP)
    i['baseline'] = np.where(i.reform_group == 'fpuc', 'baseline', 'fpuc_net')
    
"""
### Calculate % changes from relevant baselines
"""

POV_RATES_KEYS = ['year', 'age_group', 'race']  # Plus reform/baseline
POV_GAP_INEQ_KEYS = ['year']
BASELINES = ['baseline', 'fpuc_net']

pov_rates_baselines = pov_rates[pov_rates.reform.isin(
    BASELINES)][POV_RATES_KEYS + ['reform', 'pov']]
pov_rates_baselines.rename(columns={'reform': 'baseline',
                                    'pov': 'baseline_pov'}, inplace=True)
pov_rates2 = pov_rates[pov_rates.reform != 'baseline'].merge(
    pov_rates_baselines, on=POV_RATES_KEYS + ['baseline'])
pov_rates2['pov_pc'] = 100 * (pov_rates2.pov / pov_rates2.baseline_pov - 1)

pov_gap_ineq_baselines = pov_gap_ineq[
    pov_gap_ineq.reform.isin(BASELINES)][
    POV_GAP_INEQ_KEYS + ['reform', 'pov_gap_b', 'gini']]
pov_gap_ineq_baselines.rename(columns={'reform': 'baseline',
                                       'pov_gap_b': 'baseline_pov_gap_b',
                                       'gini': 'baseline_gini'}, inplace=True)
pov_gap_ineq2 = pov_gap_ineq[pov_gap_ineq.reform != 'baseline'].merge(
    pov_gap_ineq_baselines, on=POV_GAP_INEQ_KEYS + ['baseline'])
pov_gap_ineq2['pov_gap_pc'] = 100 * (pov_gap_ineq2.pov_gap_b /
                                     pov_gap_ineq2.baseline_pov_gap_b - 1)
pov_gap_ineq2['gini_pc'] = 100 * (pov_gap_ineq2.gini /
                                  pov_gap_ineq2.baseline_gini - 1)

## Charts

# Colors from https://material.io/design/color/the-color-system.html
BLUE = '#1976D2'
GRAY = '#BDBDBD'
RED = '#C62828'
LIGHT_BLUE = '#64B5F6'

COLOR_MAP = {
    '$600 per week UI': GRAY,
    'Extend $600 per week': GRAY,
    'Payroll tax cut': RED,
    'Payment to everyone': BLUE,
    'Payment to adults': LIGHT_BLUE
}

def line_graph(df, group, y, yaxis_title, title,
               x='year', color='reform_display', xaxis_title=''):
    """Style for line graphs.
    
    Arguments
        df: DataFrame with data to be plotted.
        group: Reform group, either 'fpuc' (against baseline), or
            'fpuc2' (against FPUC baseline).
        y: Column name to plot on the y axis.
        yaxis_title: y axis title.
        title: Plot title.
        x: The column name for the x axis. Defaults to 'year'.
        color: The string representing the column to show different colors of.
            Defaults to 'reform_display'.
        xaxis_title: x axis title. Defaults to '' (since the year is obvious).
    
    Returns
        Nothing. Shows the plot.
    """
    df = df[df.reform_group == group]
    if group == 'fpuc':
        yaxis_title += ' from baseline'
        title += ' from baseline'
    else:
        yaxis_title += ' from Apr-Jul FPUC baseline'
        title += ' moving forward'
    is_pc = y[-3:] == '_pc'
    if is_pc:
        df = df.round(2)
    fig = px.line(df, x=x, y=y, color=color, color_discrete_map=COLOR_MAP)
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        font=dict(family='Roboto'),
        hovermode='x',
        plot_bgcolor='white',
        legend_title_text='' 
    )
    if y == 'pov_gap_b':
        fig.update_layout(yaxis_tickprefix='$', yaxis_ticksuffix='B')
    elif y == 'pov_rate' or is_pc:  # Rate or percent changes.
        fig.update_layout(yaxis_ticksuffix='%')
    if is_pc:
        # Calculate a range to show so that the 0% zeroline is visible.
        ymin = df[y].min()
        ymax = df[y].max()
        if ymax < 0:
            ymax = 0
        yrange = ymax - ymin
        ymin_vis = ymin - 0.1 * yrange
        ymax_vis = ymax + 0.1 * yrange
        fig.update_yaxes(zeroline=True, zerolinewidth=0.5, 
                         zerolinecolor='lightgray',
                         range=[ymin_vis, ymax_vis])

    fig.update_traces(mode='markers+lines', hovertemplate=None)

    fig.show()
