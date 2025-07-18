CLASS z_humaneval_025 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.

    CLASS-METHODS factorize
      IMPORTING
        VALUE(n)      TYPE i
      RETURNING
        VALUE(result) TYPE int4_table.

  PRIVATE SECTION.
ENDCLASS.

CLASS z_humaneval_025 IMPLEMENTATION.

  METHOD factorize.
    DATA: i      TYPE i VALUE 2,
          fact   TYPE int4_table,
          sqrt_n TYPE i,
          temp_n TYPE i.

    temp_n = n.
    sqrt_n = floor( sqrt( temp_n ) ).

    WHILE i <= sqrt_n.
      WHILE temp_n MOD i = 0.
        APPEND i TO fact.
        temp_n = temp_n / i.
      ENDWHILE.
      i = i + 1.
      sqrt_n = floor( sqrt( temp_n ) ).
    ENDWHILE.

    IF temp_n > 1.
      APPEND temp_n TO fact.
    ENDIF.

    result = fact.
  ENDMETHOD.

ENDCLASS.
