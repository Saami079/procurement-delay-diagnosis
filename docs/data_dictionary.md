# Data Dictionary

## Stage-Level Dataset

| Column | Description |
|------|------------|
| Request_ID | Unique identifier for each request |
| Stage | Name of workflow stage |
| Stage_Order | Sequence order of stage |
| Role | Role responsible for approval |
| Department_Stage | Department handling the stage |
| Department_Requesting | Department initiating the request |
| Request_Type | Type of procurement request |
| Priority | Priority level (Low, Medium, High) |
| Vendor_Type | Internal or External vendor |
| Request_Amount | Monetary value of request |
| Complexity_Score | Derived complexity indicator |
| SLA_Hours | Allowed turnaround time |
| System_Load | Simulated operational load |
| Start_Time | Stage start time |
| End_Time | Stage end time |
| Processing_Time | Active processing time |
| Waiting_Time | Time before processing begins |
| Total_Stage_Time | Processing + waiting |
| Rework_Flag | Indicates rework stage |
| Is_High_Value_Request | High value flag |
| Is_High_Complexity | High complexity flag |

---

## Request-Level Dataset

| Column | Description |
|------|------------|
| Request_ID | Unique identifier |
| Request_Start | Start time of request |
| Request_End | End time of request |
| Total_Processing | Sum of processing time |
| Total_Waiting | Sum of waiting time |
| SLA_Hours | SLA threshold |
| Request_Type | Type of request |
| Priority | Priority level |
| Department_Requesting | Origin department |
| Vendor_Type | Vendor category |
| Request_Amount | Monetary value |
| Complexity_Score | Complexity level |
| System_Load | Operational load |
| Num_Stages | Number of stages |
| Total_TAT | Total turnaround time |
| Delay_Ratio | TAT / SLA |
| SLA_Breach_Hours | Excess time over SLA |
| Delayed_Flag | Target variable (0 or 1) |
| Is_High_Value_Request | High value flag |
| Is_High_Complexity | High complexity flag |
| Bottleneck_Stage | Stage with max delay |
| Max_Stage_Delay | Maximum stage delay |