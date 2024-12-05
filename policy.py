import numpy as np
import pandas as pd
import datetime as dt
from scipy.optimize import root_scalar

class Policy:
    """
    The Policy class models the behavior of an insurance policy, including the initialization 
    and calculation of cash flows and fund-related matrices.

    There are five core matrices:
        - "idx_like": Time-dependent features (Year, Anniversary, Age)
        - "fund": Values related to fund returns
        - "cf": Cash flows of the policy
        - "elig_mat": Eligibility status for phases and benefits
        - "pvs": Present values of claims and other metrics

    Major methods:
        - self._initialize(): Initializes sheets at t = 0
        - self._iter_init(): Iteratively calculates cash flows for t > 0
        - get_results(): Returns results or exports to Excel
        - update_params(): Updates parameters and re-initializes
        - reconcile(): Reconciles based on user-specified fund returns
    """

    # Class-level constants for sheet names, column definitions and output path
    sheet_names = ["idx_like", "cf", "elig_mat", "fund", "pvs"]
    columns = [
        # idx_like
        ["Year", "Anniversary", "Age"],

        # cf
        [
            "Contribution", "AV Pre-Fee", "Fund1 Pre-Fee", "Fund2 Pre-Fee", "M&E/Fund Fees",
            "AV Pre-Withdrawal", "Fund1 Pre-Withdrawal", "Fund2 Pre-Withdrawal",
            "Withdrawal Amount", "AV Post-Withdrawal", "Fund1 Post-Withdrawal",
            "Fund2 Post-Withdrawal", "Rider Charge", "AV Post-Charges",
            "Fund1 Post-Charges", "Fund2 Post-Charges", "Death Payments",
            "AV Post-Death Claims", "Fund1 Post-Death Claims", "Fund2 Post-Death Claims",
            "Fund1 Post-Rebalance", "Fund2 Post-Rebalance", "ROP Death Base",
            "NAR Death Claims", "Death Benefit Base", "Withdrawal Base",
            "Withdrawal Amount", "Cumulative Withdrawal", "Maximum Annual Withdrawal",
            "Maximum Annual Withdrawal Rate"
        ],

        # elig_mat
        [
            "Eligible Step-Up", "Growth Phase", "Withdrawal Phase",
            "Automatic Periodic Benefit Status", "Last Death"
        ],

        # fund
        ["Fund1 Return", "Fund2 Return", "Rebalance Indicator", "DF"],

        # pvs
        ["qx", "Death Claims", "Withdrawal Claims", "Rider Charges"]
    ]
    path = "./policy_results.xlsx"

    def __init__(self, params):
        """
        Initializes the Policy class with given parameters and sets up core matrices.

        Parameters:
            params (dict): Initial parameters for the policy.
        """
        for key, value in params.items():
            setattr(self, "_" + key, value)  # Set private attributes
            self._create_property(key)       # Create public properties

        # Initialize matrices during __init__
        self._idx_like, self._cf, self._elig_mat, self._fund, self._pvs = self._initialize()

    def get_results(self, to_excel=False, path=path):
        """
        Returns the generated sheets after calculations, optionally exporting to Excel.

        Parameters:
            to_excel (bool): Whether to export the results to an Excel file.
            path (str): The file path for exporting to Excel.

        Returns:
            DataFrame: Concatenated DataFrame of all sheets.
        """
        self._iter_full()  # Run the full iteration
        sheet_dfs = pd.DataFrame()
        for name, col in zip(self.sheet_names, self.columns):
            sheet = getattr(self, "_" + name)
            sheet_df = pd.DataFrame(sheet, columns=col)
            sheet_dfs = pd.concat([sheet_dfs, sheet_df], axis=1)

        if to_excel:
            sheet_dfs.to_excel(path, index=False)

        return sheet_dfs

    def update_params(self, new_params, to_excel=False, path=path):
        """
        Updates parameters and re-initializes matrices.

        Parameters:
            new_params (dict): New parameter values to update.
            to_excel (bool): Whether to export the updated results to Excel.
            path (str): The file path for exporting to Excel.

        Returns:
            DataFrame: Updated results.
        """
        for key, value in new_params.items():
            if hasattr(self, "_" + key):
                setattr(self, "_" + key, value)

        # Re-initialize matrices and get updated results
        self._idx_like, self._cf, self._elig_mat, self._fund, self._pvs = self._initialize()
        return self.get_results(to_excel, path)

    def reconcile(self, fund_ret, to_excel=False, path=path):
        """
        Reconciles the results based on a specified fund return.

        Parameters:
            fund_ret (list): User-specified returns for Fund2.
            to_excel (bool): Whether to export the reconciled results to Excel.
            path (str): The file path for exporting to Excel.

        Returns:
            DataFrame: Reconciled results.
        """
        self._idx_like, self._cf, self._elig_mat, self._fund, self._pvs = self._initialize(reconcile=True, fund_ret=fund_ret)
        return self.get_results(to_excel, path)

    def _create_property(self, key):
        """
        Creates a property for the given attribute name.

        Parameters:
            key (str): The attribute name.
        """
        def getter(self):
            return getattr(self, "_" + key)

        def setter(self, value):
            setattr(self, "_" + key, value)

        setattr(self.__class__, key, property(getter, setter))

    def _initialize(self, reconcile=False, fund_ret=None):
        """
        Initializes core matrices and calculates initial values.

        Parameters:
            reconcile (bool): Whether to reconcile using provided fund returns.
            fund_ret (list): Fund2 return values for reconciliation.

        Returns:
            list: Initialized matrices.
        """
        # initialize index-like columns
        idx_like = np.zeros((self.n_periods_inc_start,3),dtype=object)
        idx_like[:,0] = range(self.n_periods_inc_start)
        idx_like[:,1:] = [(dt.datetime.fromisoformat(str(int(self.start_date[:4]) + i) + "-" + self.start_date[5:]),60 + i) for i in idx_like[:,0]]

        # initialize fund return 
        fund = np.zeros((self.n_periods_inc_start,4))
        fund[:,-1] = np.power(1 + self.risk_free, -1*idx_like[:,0])
        fund[1:,0] = self.risk_free

        if reconcile:
            fund[1:,1] = fund_ret
        else:
            fund[1:,1] = np.exp(np.log(1 + self.risk_free) - 0.5 * self.volatility ** 2 + self.volatility * np.random.standard_normal(self.n_periods_inc_start - 1)) - 1

        # initialize cashflow
        cf = np.zeros((self.n_periods_inc_start,30))
        cf[0,2:4] = self.initial_prem * 0.16, self.initial_prem * 0.64
        cf[0,1] = cf[0,2:4].sum()
        cf[0,5] = cf[0,:2].sum() - cf[0,4]
        self._fund_ratio(cf,0,6)
        cf[0,9] = cf[0,5]
        self._fund_ratio(cf,0,10)
        self._fund_diff(cf,0,13)
        self._fund_ratio(cf,0,14)
        self._fund_diff(cf,0,17)
        self._fund_ratio(cf,0,18)
        cf[0,20] = np.where(int(fund[0,2]) == 1, cf[0,17] * self.rebalance_target, cf[0,18])
        self._fund_diff(cf,0,21,(1,8)) 
        cf[0,23] = max(0,cf[0,16] - cf[0,13])
        cf[0,[22,24,25]] = self.initial_prem
        cf[0,[8,27]] = cf[0,26]

        # initialize eligibility matrix
        elig_mat = np.zeros((self.n_periods_inc_start,5),dtype=bool)
        elig_mat[1:,1] = (idx_like[1:,-1] <= self.age_first_wd) & (idx_like[1:,-1] <= self.age_annu_com) & (idx_like[1:,-1] < self.age_death)
        elig_mat[1:,0] = (idx_like[1:,0] <= self.step_up_years) & (elig_mat[1:,1] == 1)
        elig_mat[1:,-1] = (idx_like[1:,-1] == self.age_death)

        # initialize PV
        pvs = np.zeros((self.n_periods_inc_start,4))
        pvs[:,0] = self.mort_rate
        pvs[0,1] = cf[0,23]

        return [idx_like, cf, elig_mat, fund, pvs]

    def _iter_init(self, row):
        """
        Iteratively calculates cash flows for each period t > 0.

        Parameters:
            row (int): The row index for the calculation.
        """
        idx_like = self._idx_like
        cf = self._cf
        elig_mat = self._elig_mat
        fund = self._fund
        pvs = self._pvs

        # initialize values that do not depend on wda
        elig_mat[row,2] = (
            ((idx_like[row,-1] > self.age_first_wd) | (idx_like[row,-1] > self.age_annu_com)) &
            (cf[row-1,17] > 0) &
            (idx_like[row,-1] < self.age_death)
        )
        elig_mat[row,3] = np.where(
            idx_like[row,-1] >= self.age_death, False, 
            np.where((elig_mat[row-1,2] == 1) & (int(cf[row-1,17]) == 0), True, elig_mat[row-1,3]))

        fund[row,2] = elig_mat[row,2:4].sum()

        cf[row,[2,3]] = cf[row-1,[20,21]] * (1 + fund[row,[0,1]])
        cf[row,1] = cf[row,2:4].sum()
        cf[row,4] = cf[row-1,17] * (self.m_n_e + self.fund_fee)
        cf[row,5] = max(0,cf[row,:2].sum() - cf[row,4])
        self._fund_ratio(cf,row,6)
        cf[row,22] = cf[row-1,22] * (1 - pvs[row,0])
        cf[row,16] = np.where(elig_mat[row,1:].sum() == 0, 0, cf[row - 1,[22,24]].max() * pvs[row,0])
        cf[row,29] = np.where(
            elig_mat[row,1] == 1, 0, 
            np.where(idx_like[row,-1] > self.maw_age4, self.maw_rate4,
            np.where(idx_like[row,-1] > self.maw_age3, self.maw_rate3,
            np.where(idx_like[row,-1] > self.maw_age2, self.maw_rate2,
            np.where(idx_like[row,-1] > self.maw_age1, self.maw_rate1,0)))))

        # continue on the calculation after finding wda
        wda_bar = root_scalar(self._f, args=(row),bracket=[-0.01,15000]).root
        cf[row,[8,26]] = wda_bar
        cf[row,[9,12,13,17,25,28]] = self._func_for_opt(row,x=wda_bar)
        self._fund_ratio(cf, row,10)
        self._fund_ratio(cf, row,14)
        self._fund_ratio(cf, row,18)
        cf[row,20] = np.where(int(fund[row,2]) == 1, cf[row,17] * self.rebalance_target, cf[row,18])
        self._fund_diff(cf, row,21,(1,8)) 
        cf[row,23] = max(0,cf[row,16] - cf[row,13])
        cf[row,24] = max(0, cf[row-1,24] * (1-pvs[row,0]) + cf[row,0] - cf[row,4] - cf[row-1, 8] - cf[row,12])
        cf[row,27] = cf[:row+1,26].sum()
        pvs[row,1] = cf[row,23]
        pvs[row,2] = max(cf[row,26] - cf[row-1, 17], 0)
        pvs[row,3] = cf[row, 12]

        self.cf = cf
        self.elig_mat = elig_mat
        self.fund = fund
        self.pvs = pvs

    def _iter_full(self):
        """Iterates over all periods t > 0 to compute cash flows."""
        for i in range(1,self.n_periods_inc_start):
            self._iter_init(i)

    def _fund_ratio(self, cashflow, row, num):
        """handy function for calculation"""
        cashflow[row, num] = np.where(cashflow[row, num - 1] == 0, 0, cashflow[row, num - 4] * cashflow[row, num - 1] / cashflow[row, num - 5])
        cashflow[row, num + 1] = np.where(cashflow[row, num - 1] == 0, 0, cashflow[row, num - 3] * cashflow[row, num - 1] / cashflow[row, num - 5])

    def _fund_diff(self, cashflow, row, num, offset=(1,4)):
        """another handy function for calculation"""
        cashflow[row,num] = cashflow[row, num-offset[1]] - cashflow[row, num-offset[0]]

    def _func_for_opt(self,row,x=0):
        """
        Computes intermediate steps for the withdrawal amount calculation.

        avpw - AV Post-Withdrawal

        rc - Rider Charge

        avpc - AV Post-Charge

        avpd - AV Post-Death Claims

        wdb - Withdrawal Base

        maw - Maximum Annual Withdrawal

        Parameters:
            row (int): The row index for the calculation.
            x (float): The withdrawal amount.

        Returns:
            list: Calculated values for optimization.

        """
        cf = self._cf
        elig_mat = self._elig_mat
        pvs = self._pvs

        avpw = max(0, cf[row,5] - x)
        rc = avpw * self.rider_charge
        avpc = avpw - rc
        avpd = max(avpc - cf[row,16],0)
        wdb = max(
            np.where(elig_mat[row,1] == 1, avpd, 0), 
            cf[row-1,25] * (1 - pvs[row,0]) + cf[row,0],
            np.where(elig_mat[row,0] == 1, cf[row-1,25] * (1 - pvs[row,0]) * (1 + self.step_up) + cf[row,0] - cf[row,4] - rc, 0))
        maw = wdb * cf[row,29]
        return [avpw, rc, avpc, avpd, wdb, maw]

    def _f(self,x_bar,row):
        """
        Target function for root finding.

        Parameters:
            x_bar (float): Withdrawal amount guess.
            row (int): The row index for the calculation.

        Returns:
            float: Difference between calculated withdrawal and guess.
        """
        elig_mat = self._elig_mat
        wdb, maw = self._func_for_opt(row,x_bar)[-2:]
        wda = np.where(elig_mat[row,2] == 1,self.withdraw_rate * wdb, np.where(elig_mat[row,3] == 1, maw, 0))
        return (wda - x_bar)

if __name__ == "__main__":
    params = {
        "start_date": "2016_08-01",
        "start_age": 60,
        "n_periods_inc_start": 41,
        "step_up": 0.06,
        "step_up_years": 10,
        "rider_charge": 0.0085,
        "initial_prem": 100000,
        "age_first_wd": 70,
        "age_annu_com": 80,
        "age_death": 100,
        "mort_rate": 0.005,
        "withdraw_rate": 0.03,
        "rebalance_target": 0.2,
        "maw_age1": 59.5,
        "maw_age2": 65,
        "maw_age3": 76,
        "maw_age4": 80,
        "maw_rate1": 0.04,
        "maw_rate2": 0.05,
        "maw_rate3": 0.06,
        "maw_rate4": 0.07,
        "m_n_e": 0.014,
        "fund_fee": 0.0015,
        "risk_free": 0.03,
        "volatility": 0.16
    }

    new_params = {
        "risk_free": 0.04,
    }

    fund_2_ret = [
        0.0331309375509368, 0.03223445847033, -0.0722564943428682, -0.0130506160959082,
        0.0266602837763219, 0.114682739928356, 0.107268747871032, -0.197768038754401,
        0.0104461137799554, -0.0301178326813814, -0.0081240701360142, 0.14898094632304,
        0.147539034984855, 0.209336973944666, -0.193658099664564, -0.0685566662914373,
        0.177687904207936, -0.123674679240838, -0.431728869546863, 0.0184931835922371,
        -0.192760170624528, -0.123093753907167, 0.239231279927097, -0.114466826423129,
        0.0486959231520778, -0.0623862921966919, -0.0463928877613298, -0.0104719814341985,
        -0.212071909225804, -0.0395158295089945, -0.0452107187472227, -0.281251975120906,
        -0.0287241973551424, -0.16574878574331, 0.265093341057925, 0.130806264008506,
        -0.175936061311957, 0.103488216669138, 0.0966181142832458, -0.229280184254836
    ]

    policy = Policy(params)
    results = policy.reconcile(fund_ret=fund_2_ret)
    print(results)





