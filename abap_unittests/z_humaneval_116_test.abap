CLASS ltcl_test_sort_array DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_sort_array IMPLEMENTATION.

  METHOD test_case_1.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    APPEND 1 TO input.
    APPEND 5 TO input.
    APPEND 2 TO input.
    APPEND 3 TO input.
    APPEND 4 TO input.

    APPEND 1 TO expected.
    APPEND 2 TO expected.
    APPEND 4 TO expected.
    APPEND 3 TO expected.
    APPEND 5 TO expected.

    actual = z_humaneval_116=>sort_array( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    APPEND -2 TO input.
    APPEND -3 TO input.
    APPEND -4 TO input.
    APPEND -5 TO input.
    APPEND -6 TO input.

    APPEND -4 TO expected.
    APPEND -2 TO expected.
    APPEND -6 TO expected.
    APPEND -5 TO expected.
    APPEND -3 TO expected.

    actual = z_humaneval_116=>sort_array( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    APPEND 1 TO input.
    APPEND 0 TO input.
    APPEND 2 TO input.
    APPEND 3 TO input.
    APPEND 4 TO input.

    APPEND 0 TO expected.
    APPEND 1 TO expected.
    APPEND 2 TO expected.
    APPEND 4 TO expected.
    APPEND 3 TO expected.

    actual = z_humaneval_116=>sort_array( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    actual = z_humaneval_116=>sort_array( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    APPEND 2 TO input.
    APPEND 5 TO input.
    APPEND 77 TO input.
    APPEND 4 TO input.
    APPEND 5 TO input.
    APPEND 3 TO input.
    APPEND 5 TO input.
    APPEND 7 TO input.
    APPEND 2 TO input.
    APPEND 3 TO input.
    APPEND 4 TO input.

    APPEND 2 TO expected.
    APPEND 2 TO expected.
    APPEND 4 TO expected.
    APPEND 4 TO expected.
    APPEND 3 TO expected.
    APPEND 3 TO expected.
    APPEND 5 TO expected.
    APPEND 5 TO expected.
    APPEND 5 TO expected.
    APPEND 7 TO expected.
    APPEND 77 TO expected.

    actual = z_humaneval_116=>sort_array( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    APPEND 3 TO input.
    APPEND 6 TO input.
    APPEND 44 TO input.
    APPEND 12 TO input.
    APPEND 32 TO input.
    APPEND 5 TO input.

    APPEND 32 TO expected.
    APPEND 3 TO expected.
    APPEND 5 TO expected.
    APPEND 6 TO expected.
    APPEND 12 TO expected.
    APPEND 44 TO expected.

    actual = z_humaneval_116=>sort_array( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    APPEND 2 TO input.
    APPEND 4 TO input.
    APPEND 8 TO input.
    APPEND 16 TO input.
    APPEND 32 TO input.

    APPEND 2 TO expected.
    APPEND 4 TO expected.
    APPEND 8 TO expected.
    APPEND 16 TO expected.
    APPEND 32 TO expected.

    actual = z_humaneval_116=>sort_array( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

ENDCLASS.