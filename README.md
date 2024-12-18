# Task Management App

This project is a Task Management App built using Streamlit, Pandas, SQLAlchemy, and various visualization libraries. The app provides functionalities for user authentication, data management, and visualization. It supports two types of users: employees and managers, each with different levels of access and functionality.

## Features

### User Authentication
- **Login**: Users can log in with their username and password.
- **Role-based Access Control**: Different pages and functionalities are available based on the user's role (employee or manager).

### Data Management
- **Fetch Table Data**: Users can fetch and display data from various tables in the database.
- **Upload Evidence**: Employees can upload evidence images and update the status of equipment tests.
- **Manage Employees**: Managers can add, update, and delete employee records.

### Data Visualization
- **Custom Plot Generator**: Users can generate custom plots (line chart, bar chart, scatter plot, histogram, box plot) based on selected columns and plot settings.
- **Analysis Page**: Managers can analyze data and visualize the number of tests completed by each employee.

## Technologies Used

- **Backend**:
  - Python
  - Streamlit
  - SQLAlchemy
  - pyodbc

- **Frontend**:
  - Streamlit

- **Data Visualization**:
  - Altair
  - Plotly
  - Matplotlib
  - Seaborn

## Getting Started

### Prerequisites

- Python 3.8 or higher
- SQL Server (or any other supported database)

### Installation

1. Clone the repository:
   
2. Navigate to the project directory:
   
3. Install the required dependencies:
   
4. Update the database connection string in the code:
   
5. Run the Streamlit app:
   
## Usage

### Login
- Open the app in your browser.
- Enter your username and password to log in.

### Employee Page
- Fetch and display data from various tables.
- Upload evidence images and update the status of equipment tests.

### Manager Page
- Fetch and display data from various tables.
- Analyze data and visualize the number of tests completed by each employee.
- Manage employee records (add, update, delete).

### Custom Plot Generator
- Select plot type, columns, and plot settings to generate custom plots.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.

## License

You can use it anywhere

