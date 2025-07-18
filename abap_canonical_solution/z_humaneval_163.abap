CLASS z_humaneval_163 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: generate_integers
      IMPORTING
        a             TYPE i
        b             TYPE i
      RETURNING
        VALUE(result) TYPE int4_table.

  PRIVATE SECTION.

ENDCLASS.

CLASS z_humaneval_163 IMPLEMENTATION.

  METHOD generate_integers.
    DATA: lower TYPE i,
          upper TYPE i,
          i     TYPE i.

    lower = nmax( val1 = 2 val2 = nmin( val1 = a val2 = b ) ).
    upper = nmin( val1 = 8 val2 = nmax( val1 = a val2 = b ) ).

    i = lower.
    WHILE i <= upper.
      IF i MOD 2 = 0.
        result = VALUE #( BASE result ( i ) ).
      ENDIF.
      i = i + 1.
    ENDWHILE.
  ENDMETHOD.

ENDCLASS.