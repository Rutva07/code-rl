#include<bits/stdc++.h>

using namespace std;

typedef long long ll;

int n,k;

ll sum;

int main()

{

    scanf("%d%d",&n,&k);

    if(k==0)

    {

        printf("%lld\n",(ll)n*n);

        return 0;

    }

    int x=1;

    for(int i=k+1;i<=n;i++)

    {

        sum+=(ll)n/i*x;

        if(n%i>=k)

            sum+=n%i-k+1;

        x++;

    }

    printf("%lld\n",sum);

}