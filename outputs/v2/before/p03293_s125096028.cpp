#include <stdio.h>

#include <stdlib.h>

#include<iostream>

#include<vector>

#include<cmath>

#include<cctype>

#include<algorithm>

#include<cstring>

#include<queue>

#include<map>

#include<set>

#define INF 0x3f3f3f3f

#define MAXN 1e16

#define DEBUG 0



using namespace std;

typedef long long ll;



string a,b;

int main()

{

    #if DEBUG

    freopen("in.txt","r",stdin);

    #endif // DEBUG

    cin>>a>>b;



    int ok=0;

    for(int i=0;i<a.length();i++)

    {

        if(a!=b)

        {

            string tem=b.substr(0,1);



            b.erase(b.begin());

            b=b+tem;

        }

        else {ok=1;printf("Yes");break;}

    }



    if(ok==0) printf("No");

}


