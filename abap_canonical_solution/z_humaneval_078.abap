CLASS z_humaneval_078 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS: hex_key
      IMPORTING
        VALUE(num)    TYPE string
      RETURNING
        VALUE(result) TYPE i.

ENDCLASS.

CLASS z_humaneval_078 IMPLEMENTATION.

  METHOD hex_key.
    DATA: primes TYPE string_table,
          total  TYPE i,
          char   TYPE c LENGTH 1,
          i      TYPE i.

    primes = VALUE #( ( |2| ) ( |3| ) ( |5| ) ( |7| ) ( |B| ) ( |D| ) ).
    total = 0.

    DO strlen( num ) TIMES.
      i = sy-index - 1.
      char = num+i(1).
      READ TABLE primes TRANSPORTING NO FIELDS WITH KEY table_line = char.
      IF sy-subrc = 0.
        total = total + 1.
      ENDIF.
    ENDDO.

    result = total.
  ENDMETHOD.

ENDCLASS.