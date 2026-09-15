customers = [
    {
        "name": "Alice Smith",
        "monthly_income": 5000,
        "monthly_debt": 1500,  # Standard case (30% DTI)
    },
    {
        "name": "Bob Jones",
        "monthly_income": 3000,
        "monthly_debt": 1800,  # High debt case (60% DTI)
    },
    {
        "name": "Charlie Brown",
        "monthly_income": 12000,
        "monthly_debt": 2000,  # High income, low debt case (16.6% DTI)
    },
    {
        "name": "Diana Prince",
        "monthly_income": 4500,
        "monthly_debt": 0,  # Zero debt case (0% DTI)
    },
    {
        "name": "Evan Wright",
        "monthly_income": 0,
        "monthly_debt": 500,  # Edge case: Zero income (Should handle DivisionByZero)
    },
]


def calculatedeptratio(datalist):
    print(f"{'Name':<10}|{'Income':<14}|{'Debts':<14}|{'DTI Ratio'}")
    print("-" * 55)


for customer in customers:
    name = customer.get("name", "unknown")
    income = customer.get("monthly_income", 0)
    debts = customer.get("monthly_debt", 0)

    if income > 0:
        dti_ratio = (debts / income) * 100
        dti_str = f"{dti_ratio:.1f}%"
    else:
        dti_str = "N/A(No Income)"

    print(f"{name:<10}|${income:<14,}|${debts:<14,}|{dti_str}")


calculatedeptratio(customers)
