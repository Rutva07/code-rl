#include <iostream>

#include <string>

#include <vector>



using namespace std;



int main(void) {

    string s, t;

    cin >> s;



    for (char c: s) {

        if (c=='0' || c=='1') {

            t.push_back(c);

        } else if (c=='B' && t.size()) {

            t.erase(t.end()-1);

        }

    }

    cout << t << endl;

    return 0;

}
