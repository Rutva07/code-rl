#include<stdio.h>

int main()

{

    int n, k;

    scanf("%d %d", &n, &k);

    int ans = 0;

    for (int i = 0; i < n; i++) {

        int a;

        scanf("%d", &a);

        if (a >= k) {

            ans++;

        }

    }

    printf("%d\n", ans);

    return 0;

}