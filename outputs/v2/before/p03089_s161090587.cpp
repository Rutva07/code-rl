#include<iostream>

#include<stdio.h>

#include<cstring>

#include<cmath>

#include<algorithm>

#include<vector>

#include<map>

#include<queue>

#define ll long long

int pos(int n)

{

    if(n<0)

        return 0;

    else

        return n;

}

using namespace std;

int main()

{

    int n;

    cin>>n;

    int a[n],b[n];

    int cnt=0;

    for(int i=0; i<n; i++)

    {

        cin>>a[i];

    }

    if(a[0]>1)

    {

        cout<<-1<<endl;

    }

    else

    {

        int nn=n;;

        for(int i=0; i<nn; i++)

        {

            for(int j=n-1; j>=0; j--)

            {

                if(a[j]==j+1)

                {

                    b[cnt++]=a[j];

                    for(int k=j; k<n-1; k++)

                    {

                        a[k]=a[k+1];

                    }

                    n--;

                    break;

                }

            }

        }

        if(cnt!=nn)

        {

            cout<<-1<<endl;

        }

        else

        {

            for(int i=cnt-1; i>=0; i--)

            {

                cout<<b[i]<<endl;

            }

        }

    }

    return 0;

}
