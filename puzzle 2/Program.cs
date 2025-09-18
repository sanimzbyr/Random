using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

class Program
{
    static List<int> bestDigits = new List<int>();
    static int bestLen = 0;

    static void Main()
    {
        DFS(new List<int>(), BigInteger.Zero, 0);

        if (bestLen == 0)
        {
            Console.WriteLine("No solution");
            return;
        }

        Console.WriteLine(DigitsToHex(bestDigits));
    }

    // Depth-first search over hex digits with modular constraint:
    // For k+1: (16*N + d) % (k+1) == (k+1) - 1
    static void DFS(List<int> digits, BigInteger N, int k)
    {
        // If current length is odd and the middle digit is 3, record as best
        if (k > bestLen && (k & 1) == 1 && digits[k / 2] == 3)
        {
            bestLen = k;
            bestDigits = new List<int>(digits);
        }

        int kp1 = k + 1;
        int need = kp1 - 1;
        int baseMod = (int)((16 * (N % kp1)) % kp1);
        int r = need - baseMod;
        r %= kp1;
        if (r < 0) r += kp1;

        // Candidate digits in [0..15] with d ≡ r (mod kp1)
        if (kp1 > 16)
        {
            // Unique candidate (if any)
            if (r <= 15)
                TryNext(digits, N, k, r);
        }
        else
        {
            for (int d = r; d <= 15; d += kp1)
                TryNext(digits, N, k, d);
        }
    }

    static void TryNext(List<int> digits, BigInteger N, int k, int d)
    {
        if (k == 0 && d == 0) return; // no leading zero

        digits.Add(d);
        BigInteger Nnext = N * 16 + d;

        int kp1 = k + 1;
        if ((Nnext % kp1) == kp1 - 1) // should always hold by construction
            DFS(digits, Nnext, kp1);

        digits.RemoveAt(digits.Count - 1);
    }

    static string DigitsToHex(List<int> digits)
    {
        char HexDigit(int x) => (char)(x < 10 ? '0' + x : 'A' + (x - 10));
        return new string(digits.Select(HexDigit).ToArray());
    }
}
