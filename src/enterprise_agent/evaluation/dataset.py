"""Synthetic ground-truth enterprise benchmark dataset for RAG evaluation."""

from enterprise_agent.schemas.evaluation import EvaluationSample

ENTERPRISE_BENCHMARK_DATASET: list[EvaluationSample] = [
    EvaluationSample(
        query=(
            "What is the monthly internet allowance and initial equipment reimbursement "
            "for remote employees?"
        ),
        ground_truth=(
            "Remote employees receive a monthly internet allowance of $100 and a one-time "
            "initial home office equipment reimbursement of up to $500."
        ),
        golden_contexts=[
            (
                "Section 3.1: Remote employees are eligible for a $100 monthly stipend for "
                "high-speed internet. Upon onboarding, new remote staff may submit expense "
                "receipts up to $500 for ergonomic desk equipment, monitors, and accessories."
            )
        ],
    ),
    EvaluationSample(
        query="What is the maximum nightly hotel lodging reimbursement rate?",
        ground_truth=(
            "Standard hotel lodging is reimbursed up to $250 per night in Tier 1 "
            "metropolitan areas and $175 per night in all other locations."
        ),
        golden_contexts=[
            (
                "Section 4.2: Lodging expenses are capped at $250 per night in Tier 1 "
                "metropolitan cities (e.g., New York, San Francisco, London) and $175 per night "
                "in all other destinations. Room taxes are reimbursed in addition to these rates."
            )
        ],
    ),
    EvaluationSample(
        query="When are employees permitted to book business class flights?",
        ground_truth=(
            "Employees may book business class flights only for continuous international "
            "flights exceeding 6 hours in flight time."
        ),
        golden_contexts=[
            (
                "Section 4.1: Economy class must be booked for all domestic and regional "
                "travel under 6 hours. For international flights exceeding 6 hours "
                "continuous flight time, business class booking is pre-approved."
            )
        ],
    ),
    EvaluationSample(
        query="How many days of unused paid time off can be carried over into the next year?",
        ground_truth=(
            "Employees may carry over a maximum of 5 unused PTO days into the following "
            "calendar year, which must be used before March 31."
        ),
        golden_contexts=[
            (
                "Section 5.4: Unused vacation time does not automatically accumulate. Employees "
                "are permitted to carry over up to 5 unused PTO days into the next calendar year, "
                "expiring on March 31."
            )
        ],
    ),
    EvaluationSample(
        query="When is the annual open enrollment period for employee health insurance?",
        ground_truth=(
            "Annual open enrollment for corporate health insurance occurs every year from "
            "November 1 through November 30."
        ),
        golden_contexts=[
            (
                "Section 6.2: Open enrollment for medical, dental, and vision benefits begins "
                "annually on November 1 and closes on November 30 at 11:59 PM EST."
            )
        ],
    ),
    EvaluationSample(
        query="What is the laptop and hardware refresh policy for Engineering employees?",
        ground_truth=(
            "Engineering laptops are eligible for standard hardware replacement every 3 "
            "years or upon promotion to Senior Staff Engineer."
        ),
        golden_contexts=[
            (
                "Section 7.3: Standard laptops for Engineering staff are refreshed on a "
                "36-month (3-year) cycle. Exceptions are granted for hardware failure or "
                "promotion to Senior Staff Engineer."
            )
        ],
    ),
    EvaluationSample(
        query="What are the corporate password complexity requirements and rotation intervals?",
        ground_truth=(
            "Passwords must be at least 16 characters in length, contain uppercase, lowercase, "
            "numbers, and symbols, and be rotated every 90 days."
        ),
        golden_contexts=[
            (
                "Section 8.1: Security credentials must contain at least 16 characters "
                "including uppercase letters, lowercase letters, numbers, and special symbols. "
                "Passwords expire every 90 days."
            )
        ],
    ),
    EvaluationSample(
        query="How many days of paid bereavement leave are provided for immediate family members?",
        ground_truth=(
            "Employees are provided up to 5 consecutive paid days of bereavement leave for "
            "the loss of an immediate family member."
        ),
        golden_contexts=[
            (
                "Section 5.8: Full-time employees are entitled to up to 5 consecutive days of "
                "paid bereavement leave in the event of the death of an immediate family member."
            )
        ],
    ),
    EvaluationSample(
        query="How often are internal user access controls and permissions reviewed for SOC2?",
        ground_truth=(
            "Internal access permissions and user access rights must undergo mandatory "
            "compliance audits once every quarter."
        ),
        golden_contexts=[
            (
                "Section 9.4: In accordance with SOC2 Type II compliance controls, user access "
                "to production databases and infrastructure is reviewed quarterly by the "
                "Information Security team."
            )
        ],
    ),
    EvaluationSample(
        query="What is the maximum value of gifts an employee can accept from vendors?",
        ground_truth=(
            "Employees cannot accept gifts from vendors valued in excess of $50 without "
            "explicit written approval from Legal and Compliance."
        ),
        golden_contexts=[
            (
                "Section 10.2: Employees must not accept gifts, entertainment, or gratuities "
                "valued above $50 from existing or prospective vendors without prior written "
                "authorization from Legal Compliance."
            )
        ],
    ),
]
