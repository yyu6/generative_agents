"""Enumerations used by Financial Town entities."""

from enum import Enum


class EmploymentType(str, Enum):
  SALARIED = "salaried"
  HOURLY = "hourly"
  SELF_EMPLOYED = "self_employed"


class EmploymentStatus(str, Enum):
  EMPLOYED = "employed"
  UNEMPLOYED = "unemployed"


class PayCycle(str, Enum):
  WEEKLY = "weekly"
  BIWEEKLY = "biweekly"
  MONTHLY = "monthly"


class LoanType(str, Enum):
  HOME = "home"
  EDUCATION = "education"
  PERSONAL = "personal"
  INVESTMENT = "investment"
  EMERGENCY = "emergency"


class LoanStatus(str, Enum):
  ACTIVE = "active"
  DELINQUENT = "delinquent"
  CLOSED = "closed"
  WRITTEN_OFF = "written_off"


class TransactionType(str, Enum):
  INCOME = "income"
  EXPENSE = "expense"
  HOUSING = "housing"
  LOAN_PAYMENT = "loan_payment"
  LOAN_ORIGINATION = "loan_origination"
  FRAUD_ALERT = "fraud_alert"
