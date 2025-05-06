#include <iostream>
#include <string>
using namespace std;


void showOptions() {
    cout << "\nHow can I help you today?\n";
    cout << "1. Order Status\n";
    cout << "2. Product Information\n";
    cout << "3. Contact Support\n";
    cout << "4. Exit\n";
}

int main() {
    string botName = "Chatty";
    string userName;
    int choice;

    cout << "Hello! My name is " << botName << ". I am your Customer Support Assistant.\n";

    cout << "May I know your name? ";
    cin >> userName;

    cout << "Nice to meet you, " << userName << "!\n";

    do {
        showOptions();
        cout << "\nPlease enter your choice (1-4): ";
        cin >> choice;

        switch (choice) {
            case 1:
                cout << "\n" << userName << ", to check your order status, please log into your account and go to 'My Orders'. You can also track your shipment via the tracking link sent to your email.\n";
                break;
            case 2:
                cout << "\nWe offer a wide range of products including electronics, clothing, home appliances, and more. Visit our catalog for detailed descriptions and prices.\n";
                break;
            case 3:
                cout << "\nYou can reach our support team at: support@company.com or call us at +1-800-555-1234 (9 AM to 6 PM, Mon-Sat).\n";
                break;
            case 4:
                cout << "\nThank you for contacting support, " << userName << "! If you need further assistance, we're always here to help.\n";
                break;
            default:
                cout << "\nInvalid choice. Please select a valid option.\n";
        }

    } while (choice != 4);

    return 0;
}
