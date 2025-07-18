CLASS z_humaneval_123 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: get_odd_collatz
      IMPORTING
        VALUE(n)      TYPE i
      RETURNING
        VALUE(result) TYPE int4_table.
  PROTECTED SECTION.
  PRIVATE SECTION.
ENDCLASS.

CLASS z_humaneval_123 IMPLEMENTATION.
  METHOD get_odd_collatz.
    DATA: odd_collatz TYPE int4_table,
          current_n   TYPE i.

    current_n = n.

    IF current_n MOD 2 = 0.
      CLEAR odd_collatz.
    ELSE.
      odd_collatz = VALUE #( ( current_n ) ).
    ENDIF.

    WHILE current_n > 1.
      IF current_n MOD 2 = 0.
        current_n = current_n / 2.
      ELSE.
        current_n = current_n * 3 + 1.
      ENDIF.

      IF current_n MOD 2 = 1.
        odd_collatz = VALUE #( BASE odd_collatz ( current_n ) ).
      ENDIF.
    ENDWHILE.

    SORT odd_collatz.
    result = odd_collatz.
  ENDMETHOD.
ENDCLASS.