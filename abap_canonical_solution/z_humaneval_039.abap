CLASS z_humaneval_039 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: prime_fib
      IMPORTING
        n             TYPE i
      RETURNING
        VALUE(result) TYPE i.

  PRIVATE SECTION.
    CLASS-METHODS: is_prime
      IMPORTING
        p             TYPE i
      RETURNING
        VALUE(result) TYPE abap_bool.

ENDCLASS.

CLASS z_humaneval_039 IMPLEMENTATION.

  METHOD prime_fib.
    DATA: f         TYPE int4_table,
          current_n TYPE i,
          next_fib  TYPE i.

    current_n = n.
    f = VALUE #( ( 0 ) ( 1 ) ).

    WHILE abap_true = abap_true.
      READ TABLE f INDEX lines( f ) INTO DATA(last).
      READ TABLE f INDEX lines( f ) - 1 INTO DATA(second_last).
      next_fib = last + second_last.
      APPEND next_fib TO f.

      IF is_prime( next_fib ) = abap_true.
        current_n = current_n - 1.
      ENDIF.

      IF current_n = 0.
        result = next_fib.
        EXIT.
      ENDIF.
    ENDWHILE.
  ENDMETHOD.

  METHOD is_prime.
    DATA: k      TYPE i,
          sqrt_p TYPE i.

    IF p < 2.
      result = abap_false.
      RETURN.
    ENDIF.

    sqrt_p = sqrt( p ).

    k = 1.
    DO.
      k = k + 1.
      IF k > sqrt_p OR k >= p - 1.
        EXIT.
      ENDIF.
      IF p MOD k = 0.
        result = abap_false.
        RETURN.
      ENDIF.
    ENDDO.

    result = abap_true.
  ENDMETHOD.

ENDCLASS.