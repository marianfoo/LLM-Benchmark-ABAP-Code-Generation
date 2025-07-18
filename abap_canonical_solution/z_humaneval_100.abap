CLASS z_humaneval_100 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: make_a_pile
      IMPORTING
        VALUE(n)      TYPE i
      RETURNING
        VALUE(result) TYPE int4_table.

ENDCLASS.

CLASS z_humaneval_100 IMPLEMENTATION.

  METHOD make_a_pile.
    DATA: i      TYPE i,
          stones TYPE i.

    DO n TIMES.
      i = sy-index - 1.
      stones = n + 2 * i.
      APPEND stones TO result.
    ENDDO.

  ENDMETHOD.

ENDCLASS.