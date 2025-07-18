CLASS ltc_test_select_words DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_first_case FOR TESTING,
      test_second_case FOR TESTING,
      test_third_case FOR TESTING,
      test_fourth_case FOR TESTING,
      test_fifth_case FOR TESTING,
      test_edge_case_empty FOR TESTING,
      test_edge_single_consonants FOR TESTING.

ENDCLASS.

CLASS ltc_test_select_words IMPLEMENTATION.

  METHOD test_first_case.
    DATA: result   TYPE string_table,
          expected TYPE string_table.

    APPEND 'little' TO expected.
    result = z_humaneval_117=>select_words( s = 'Mary had a little lamb' n = 4 ).

    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = expected
      msg = 'First test error'
    ).
  ENDMETHOD.

  METHOD test_second_case.
    DATA: result   TYPE string_table,
          expected TYPE string_table.

    APPEND 'Mary' TO expected.
    APPEND 'lamb' TO expected.
    result = z_humaneval_117=>select_words( s = 'Mary had a little lamb' n = 3 ).

    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = expected
      msg = 'Second test error'
    ).
  ENDMETHOD.

  METHOD test_third_case.
    DATA: result   TYPE string_table,
          expected TYPE string_table.

    result = z_humaneval_117=>select_words( s = 'simple white space' n = 2 ).

    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = expected
      msg = 'Third test error'
    ).
  ENDMETHOD.

  METHOD test_fourth_case.
    DATA: result   TYPE string_table,
          expected TYPE string_table.

    APPEND 'world' TO expected.
    result = z_humaneval_117=>select_words( s = 'Hello world' n = 4 ).

    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = expected
      msg = 'Fourth test error'
    ).
  ENDMETHOD.

  METHOD test_fifth_case.
    DATA: result   TYPE string_table,
          expected TYPE string_table.

    APPEND 'Uncle' TO expected.
    result = z_humaneval_117=>select_words( s = 'Uncle sam' n = 3 ).

    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = expected
      msg = 'Fifth test error'
    ).
  ENDMETHOD.

  METHOD test_edge_case_empty.
    DATA: result   TYPE string_table,
          expected TYPE string_table.

    result = z_humaneval_117=>select_words( s = '' n = 4 ).

    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = expected
      msg = '1st edge test error'
    ).
  ENDMETHOD.

  METHOD test_edge_single_consonants.
    DATA: result   TYPE string_table,
          expected TYPE string_table.

    APPEND 'b' TO expected.
    APPEND 'c' TO expected.
    APPEND 'd' TO expected.
    APPEND 'f' TO expected.
    result = z_humaneval_117=>select_words( s = 'a b c d e f' n = 1 ).

    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = expected
      msg = '2nd edge test error'
    ).
  ENDMETHOD.

ENDCLASS.