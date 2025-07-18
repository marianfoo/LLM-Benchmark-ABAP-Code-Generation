CLASS z_humaneval_139 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: special_factorial
      IMPORTING
        n             TYPE i
      RETURNING
        VALUE(result) TYPE int8.

ENDCLASS.

CLASS z_humaneval_139 IMPLEMENTATION.

  METHOD special_factorial.
    DATA: fact_i       TYPE int8 VALUE 1,
          special_fact TYPE int8 VALUE 1,
          i            TYPE i.

    DO n TIMES.
      i = sy-index.
      fact_i = fact_i * i.
      special_fact = special_fact * fact_i.
    ENDDO.

    result = special_fact.
  ENDMETHOD.

ENDCLASS.