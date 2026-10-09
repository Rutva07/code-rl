#include <iostream>

#include <string>

#include <vector>

using namespace std;



int main()

{

    string s;

    long long n, z=0, r = 0, g = 0, b = 0;

    cin >> n;

    cin >> s;

    for (long long i = 0; i < n; i++)

    {

        if (s[i] == 'R')

        {

            z+=g*b;

            r++;

        }

        if (s[i] == 'G')

        {

            z+=r*b;

            g++;

        }

        if (s[i] == 'B')

        {

            z+=r*g;

            b++;

        }

    }

        for (long long o = 3; o-1 < n; o += 2) {

      for (long long i = o-1; i < n; i++){

        if (s[i] != s[i-o+1] && s[i-o+1] != s[i-o/2] && s[i-o/2] !=s[i])

          z--;

        }

    }

    cout<<z<<endl;

    return 0;

}